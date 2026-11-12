"""Tests for the quality gate (geometry rules, photometrics) and the try-on service."""

import numpy as np
import pytest
from PIL import Image, ImageFilter

from conftest import make_image
from tryon_server.mock_engine import MockEngine
from virtual_tryon import quality_check as qc
from virtual_tryon.backends.local import LocalBackend
from virtual_tryon.service import TryOnService


def frontal_landmarks(w=600, h=800) -> np.ndarray:
    """Synthetic front-facing person: person's left side appears on the image's right."""
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[:, 3] = 0.99
    pts = {
        qc.NOSE: (300, 120), qc.L_SHOULDER: (380, 220), qc.R_SHOULDER: (220, 220),
        qc.L_WRIST: (420, 470), qc.R_WRIST: (180, 470), qc.L_HIP: (350, 480), qc.R_HIP: (250, 480),
        qc.L_KNEE: (350, 640), qc.R_KNEE: (250, 640), qc.L_ANKLE: (350, 780), qc.R_ANKLE: (250, 780),
    }
    for idx, (x, y) in pts.items():
        lm[idx, :2] = (x, y)
    for idx in range(33):
        if idx not in pts:
            lm[idx, :2] = (300, 300)
    return lm


def codes(report):
    return {i.code for i in report.issues}


def test_frontal_pose_passes():
    report = qc.QualityReport()
    qc.analyse_pose(frontal_landmarks(), (600, 800), "upper", report)
    assert report.ok and codes(report) == set()
    assert 0.5 < report.metrics["shoulder_torso_ratio"] < 0.7


def test_missing_hips_is_error_for_upper():
    lm = frontal_landmarks()
    lm[[qc.L_HIP, qc.R_HIP], 3] = 0.1
    report = qc.QualityReport()
    qc.analyse_pose(lm, (600, 800), "upper", report)
    assert not report.ok and "missing_hips" in codes(report)


def test_knees_needed_for_lower_only():
    lm = frontal_landmarks()
    lm[[qc.L_KNEE, qc.R_KNEE], 3] = 0.0
    upper, lower = qc.QualityReport(), qc.QualityReport()
    qc.analyse_pose(lm, (600, 800), "upper", upper)
    qc.analyse_pose(lm, (600, 800), "lower", lower)
    assert upper.ok and "missing_knees" in codes(lower)


def test_side_view_and_back_view():
    side = frontal_landmarks()
    side[qc.L_SHOULDER, 0], side[qc.R_SHOULDER, 0] = 310, 290      # shoulders almost overlap
    report = qc.QualityReport()
    qc.analyse_pose(side, (600, 800), "upper", report)
    assert "side_view" in codes(report)

    back = frontal_landmarks()
    back[[qc.L_SHOULDER, qc.R_SHOULDER], 0] = back[[qc.R_SHOULDER, qc.L_SHOULDER], 0]  # swapped sides
    report = qc.QualityReport()
    qc.analyse_pose(back, (600, 800), "upper", report)
    assert "back_view" in codes(report)


def test_photo_cut_off_at_hips_is_error():
    lm = frontal_landmarks()
    lm[[qc.L_HIP, qc.R_HIP], 1] = 740          # hips at 0.93 of an 800 px image
    lm[[qc.L_KNEE, qc.R_KNEE, qc.L_ANKLE, qc.R_ANKLE], 3] = 0.1
    report = qc.QualityReport()
    qc.analyse_pose(lm, (600, 800), "upper", report)
    assert not report.ok and "cut_off_at_hips" in codes(report)


def test_check_runs_on_model_crop():
    wide = make_image(1600, 800, noise=False)
    assert qc.model_view(wide).size == (600, 800)


def test_hand_on_torso_info():
    lm = frontal_landmarks()
    lm[qc.L_WRIST, :2] = (300, 350)
    report = qc.QualityReport()
    qc.analyse_pose(lm, (600, 800), "upper", report)
    assert "hands_on_torso" in codes(report) and report.ok


def test_photometric_checks():
    dark = qc.check_person_image(make_image(color=(15, 15, 15)), use_pose=False)
    assert "dark" in codes(dark)
    blurry = make_image(color=(128, 128, 128)).filter(ImageFilter.GaussianBlur(12))
    assert "blurry" in codes(qc.check_person_image(blurry, use_pose=False))


def test_garment_checks(garment_image):
    assert "busy_background" not in codes(qc.check_garment_image(garment_image))
    # cluttered scene: random coloured 50 px blocks (like a room behind a garment)
    rng = np.random.default_rng(1)
    blocks = rng.integers(0, 255, (12, 10, 3), dtype=np.uint8)
    cluttered = Image.fromarray(blocks).resize((500, 600), Image.NEAREST)
    assert "busy_background" in codes(qc.check_garment_image(cluttered))


def test_service_with_mock_backend(person_image, garment_image, tmp_path, monkeypatch):
    service = TryOnService(LocalBackend(MockEngine()), save_results=True)
    outcome = service.run(person_image, garment_image, "upper", steps=6, guidance=2.5, seed=1)
    assert outcome.result.image.size == outcome.prepared.person_preview.size == (768, 1024)
    assert outcome.saved is None   # mock output must never enter the results gallery


@pytest.mark.slow
def test_mediapipe_finds_no_person_in_noise():
    """Runs the real MediaPipe model (downloads ~5 MB on first run)."""
    report = qc.check_person_image(make_image(600, 800, color=(120, 160, 200)), "upper")
    assert "no_person" in codes(report) or "pose_unavailable" in codes(report)
