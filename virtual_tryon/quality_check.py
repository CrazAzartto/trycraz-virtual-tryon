"""Input-quality gate: warns the user *before* a slow GPU job if a photo is likely to fail.

This is our own computer-vision component (CatVTON does not do this):
  - Human pose estimation with MediaPipe PoseLandmarker (BlazePose, 33 body landmarks,
    runs on the laptop CPU in well under a second).
  - Geometric rules on the landmarks: is a person present, is the body region needed for the
    chosen garment type visible, is the person facing the camera, large enough in the frame,
    and is only one person present.
  - Classical image statistics: brightness (mean luminance) and sharpness (variance of the
    Laplacian, a standard focus measure).

All thresholds are heuristics chosen by us, not learned. They only produce warnings; the
user can still generate. Each check is documented next to its threshold so it can be
explained (and tuned) in the report.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from config import settings

log = logging.getLogger(__name__)

# MediaPipe BlazePose landmark indices (33-point topology)
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# Landmarks that must be visible for each CatVTON garment type
REQUIRED_LANDMARKS = {
    "upper": {"shoulders": (L_SHOULDER, R_SHOULDER), "hips": (L_HIP, R_HIP)},
    "lower": {"hips": (L_HIP, R_HIP), "knees": (L_KNEE, R_KNEE)},
    "overall": {"shoulders": (L_SHOULDER, R_SHOULDER), "hips": (L_HIP, R_HIP), "knees": (L_KNEE, R_KNEE)},
}

SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28)]

# ---- thresholds (heuristics) ----------------------------------------------
VISIBILITY_MIN = 0.5          # MediaPipe per-landmark visibility score
DARK_LUMA = 60                # mean luminance (0-255) below this -> too dark
BRIGHT_LUMA = 225             # above this -> overexposed
# Laplacian variance (computed on a 512 px version) below this -> blurry.
# Calibrated 2026-11-08 on the 20 CatVTON demo images: sharp originals scored 50-1562,
# the same images blurred with sigma=1.5 px (at 512 px) scored 2-28. 35 sits between the groups.
# Small calibration set: treat as a heuristic, re-check on our own photos in Phase 6.
BLUR_VAR = 35.0
MIN_BODY_HEIGHT_FRAC = 0.30   # visible body landmarks span < 30% of image height -> person too small
SIDE_VIEW_RATIO = 0.35        # shoulder width / torso length below this -> body turned sideways
GARMENT_FILL_RANGE = (0.12, 0.97)  # fraction of garment-image pixels that differ from the background
# Hip height as a fraction of the (3:4-cropped) image height. Calibrated 2026-11-08 on real CatVTON runs:
# photos cut off at the hips (hip_y 0.89 and 0.91) produced a grey filler below the garment hem in 3/3 runs,
# while photos with hip_y 0.43-0.50 worked. The garment needs body visible below its hem.
HIP_Y_ERROR = 0.85
HIP_Y_WARNING = 0.75


@dataclass
class QualityIssue:
    level: str      # "error" (very likely to fail) | "warning" (may degrade) | "info"
    code: str
    message: str


@dataclass
class QualityReport:
    issues: list[QualityIssue] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    landmarks: Optional[np.ndarray] = None   # (33, 4): x_px, y_px, z, visibility for the first person

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    def add(self, level: str, code: str, message: str) -> None:
        self.issues.append(QualityIssue(level, code, message))


# ---------------------------------------------------------------------------
# Image statistics (no model needed)
# ---------------------------------------------------------------------------
def _gray_512(image: Image.Image) -> np.ndarray:
    img = image.convert("L")
    scale = 512 / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BILINEAR)
    return np.asarray(img)


def brightness_and_sharpness(image: Image.Image) -> tuple[float, float]:
    gray = _gray_512(image)
    return float(gray.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _photometric_checks(image: Image.Image, report: QualityReport, role: str) -> None:
    luma, sharp = brightness_and_sharpness(image)
    report.metrics.update({"mean_luminance": round(luma, 1), "laplacian_variance": round(sharp, 1)})
    if luma < DARK_LUMA:
        report.add("warning", "dark", f"The {role} photo is quite dark; details may be lost. Use brighter, even lighting.")
    elif luma > BRIGHT_LUMA:
        report.add("warning", "bright", f"The {role} photo looks overexposed; colours may wash out.")
    if sharp < BLUR_VAR:
        report.add("warning", "blurry", f"The {role} photo looks blurry. A sharper photo gives a more detailed result.")


# ---------------------------------------------------------------------------
# Pose estimation
# ---------------------------------------------------------------------------
_landmarker = None
_landmarker_lock = threading.Lock()


def ensure_pose_model(path: Path = settings.POSE_MODEL_PATH) -> Path:
    """Download the MediaPipe pose model (~5.5 MB) on first use."""
    if path.exists() and path.stat().st_size > 0:
        return path
    import requests
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading MediaPipe pose model to %s", path)
    response = requests.get(settings.POSE_MODEL_URL, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def _get_landmarker():
    global _landmarker
    with _landmarker_lock:
        if _landmarker is None:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision
            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(ensure_pose_model())),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=3,                         # >1 so we can warn about extra people
                min_pose_detection_confidence=0.5,
            )
            _landmarker = vision.PoseLandmarker.create_from_options(options)
        return _landmarker


def detect_poses(image: Image.Image) -> list[np.ndarray]:
    """Return one (33, 4) array per detected person: x_px, y_px, z, visibility."""
    import mediapipe as mp
    rgb = np.ascontiguousarray(np.asarray(image.convert("RGB")))
    result = _get_landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    w, h = image.size
    poses = []
    for person in result.pose_landmarks:
        poses.append(np.array([[lm.x * w, lm.y * h, lm.z, lm.visibility] for lm in person], dtype=np.float32))
    return poses


def analyse_pose(landmarks: np.ndarray, image_size: tuple[int, int], garment_type: str, report: QualityReport) -> None:
    """Geometric rules on one person's landmarks (pure function: unit-testable without MediaPipe)."""
    w, h = image_size
    vis = landmarks[:, 3]
    xy = landmarks[:, :2]

    for part, (a, b) in REQUIRED_LANDMARKS[garment_type].items():
        visible = [vis[a] >= VISIBILITY_MIN, vis[b] >= VISIBILITY_MIN]
        inside = [0 <= xy[i, 0] < w and 0 <= xy[i, 1] < h for i in (a, b)]
        if not (all(visible) and all(inside)):
            report.add("error", f"missing_{part}",
                       f"Both {part} should be clearly visible for a '{garment_type}' try-on. "
                       "Use a photo that shows more of the body.")

    shoulders_ok = vis[L_SHOULDER] >= VISIBILITY_MIN and vis[R_SHOULDER] >= VISIBILITY_MIN
    hips_ok = vis[L_HIP] >= VISIBILITY_MIN and vis[R_HIP] >= VISIBILITY_MIN
    if shoulders_ok and hips_ok:
        shoulder_width = float(np.linalg.norm(xy[L_SHOULDER] - xy[R_SHOULDER]))
        torso_length = float(np.linalg.norm((xy[L_SHOULDER] + xy[R_SHOULDER]) / 2 - (xy[L_HIP] + xy[R_HIP]) / 2))
        ratio = shoulder_width / max(torso_length, 1e-6)
        report.metrics["shoulder_torso_ratio"] = round(ratio, 2)
        if ratio < SIDE_VIEW_RATIO:
            report.add("warning", "side_view",
                       "The body seems turned sideways. The model works best on front-facing photos.")
        # Facing the camera, the person's LEFT shoulder appears on the RIGHT of the image.
        if xy[L_SHOULDER, 0] < xy[R_SHOULDER, 0] and ratio >= SIDE_VIEW_RATIO:
            report.add("warning", "back_view", "The person may be facing away from the camera.")
        # Hands in front of the torso get repainted together with the garment.
        x0, x1 = sorted([xy[L_SHOULDER, 0], xy[R_SHOULDER, 0]])
        y0, y1 = min(xy[L_SHOULDER, 1], xy[R_SHOULDER, 1]), max(xy[L_HIP, 1], xy[R_HIP, 1])
        for wrist in (L_WRIST, R_WRIST):
            if vis[wrist] >= VISIBILITY_MIN and x0 < xy[wrist, 0] < x1 and y0 < xy[wrist, 1] < y1:
                report.add("info", "hands_on_torso",
                           "A hand is in front of the torso. Hands are protected, but arms crossing the body can cause artefacts.")
                break

    if hips_ok:
        hip_y = float(max(xy[L_HIP, 1], xy[R_HIP, 1]) / h)
        report.metrics["hip_y_fraction"] = round(hip_y, 2)
        if hip_y > HIP_Y_ERROR:
            report.add("error", "cut_off_at_hips",
                       "The photo is cut off at the hips. Include your body down to at least mid-thigh, "
                       "otherwise the model fills the space below the garment with grey.")
        elif hip_y > HIP_Y_WARNING:
            report.add("warning", "low_hips", "Little of the body is visible below the hips; longer garments may be cut short.")

    visible_y = xy[vis >= VISIBILITY_MIN, 1]
    if visible_y.size >= 4:
        frac = float((visible_y.max() - visible_y.min()) / h)
        report.metrics["body_height_fraction"] = round(frac, 2)
        if frac < MIN_BODY_HEIGHT_FRAC:
            report.add("warning", "small_person", "The person is small in the frame. Move closer or crop the photo.")


def model_view(image: Image.Image) -> Image.Image:
    """The part of the photo the model will actually see (CatVTON centre-crops to 3:4)."""
    from common.image_utils import center_crop_to_aspect
    return center_crop_to_aspect(image, settings.MODEL_WIDTH, settings.MODEL_HEIGHT)


def check_person_image(image: Image.Image, garment_type: str = "upper", use_pose: bool = True) -> QualityReport:
    """Checks run on model_view(image); landmarks are in that cropped image's coordinates."""
    image = model_view(image)
    report = QualityReport()
    _photometric_checks(image, report, "person")
    if not use_pose:
        return report
    try:
        poses = detect_poses(image)
    except Exception as exc:   # model download / runtime problems must not block try-on
        log.warning("Pose check unavailable: %s", exc)
        report.add("info", "pose_unavailable", "Pose check unavailable on this machine; skipped.")
        return report
    report.metrics["people_detected"] = len(poses)
    if not poses:
        report.add("error", "no_person", "No person was detected. Use a clear photo of one person, facing the camera.")
        return report
    if len(poses) > 1:
        report.add("warning", "multiple_people", "More than one person was detected; the most prominent one will be dressed.")
    report.landmarks = poses[0]
    analyse_pose(poses[0], image.size, garment_type, report)
    return report


def check_garment_image(image: Image.Image) -> QualityReport:
    """Product-style garment photos (garment alone on a plain background) work best."""
    report = QualityReport()
    _photometric_checks(image, report, "garment")
    arr = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.float32)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    background = np.median(border, axis=0)
    border_spread = float(np.abs(border - background).mean())
    fill = float((np.abs(arr - background).max(axis=2) > 40).mean())
    report.metrics.update({"background_spread": round(border_spread, 1), "garment_fill": round(fill, 2)})
    if border_spread > 30:
        report.add("warning", "busy_background",
                   "The garment photo has a busy background. A plain background (or a product photo) works best.")
    if fill < GARMENT_FILL_RANGE[0]:
        report.add("warning", "garment_small", "The garment is small in the photo. Crop closer to the garment.")
    elif fill > GARMENT_FILL_RANGE[1]:
        report.add("info", "garment_fills_frame", "The garment fills the whole photo; make sure its full outline is visible.")
    return report


def draw_skeleton(image: Image.Image, landmarks: np.ndarray) -> Image.Image:
    """Visualise detected landmarks (shown in the UI and used in the report)."""
    canvas = np.asarray(image.convert("RGB")).copy()
    thickness = max(2, image.width // 250)
    for a, b in SKELETON:
        if landmarks[a, 3] >= VISIBILITY_MIN and landmarks[b, 3] >= VISIBILITY_MIN:
            cv2.line(canvas, tuple(int(v) for v in landmarks[a, :2]), tuple(int(v) for v in landmarks[b, :2]),
                     (0, 200, 255), thickness, cv2.LINE_AA)
    for x, y, _, v in landmarks:
        if v >= VISIBILITY_MIN:
            cv2.circle(canvas, (int(x), int(y)), thickness * 2, (255, 80, 0), -1, cv2.LINE_AA)
    return Image.fromarray(canvas)
