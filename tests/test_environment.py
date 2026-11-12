"""Smoke tests: the environment has every package the app needs and config loads."""

import importlib

import pytest

REQUIRED = ["streamlit", "PIL", "numpy", "cv2", "yaml", "requests", "mediapipe", "torch", "transformers"]


@pytest.mark.parametrize("module_name", REQUIRED)
def test_required_package_importable(module_name):
    importlib.import_module(module_name)


def test_settings_load():
    from config import settings

    assert settings.PROJECT_ROOT.exists()
    assert settings.MODEL_WIDTH * 4 == settings.MODEL_HEIGHT * 3  # CatVTON's 3:4 aspect ratio
    assert set(settings.GARMENT_TYPES) == {"upper", "lower", "overall"}
