# Execution Environment

## Laptop (development + Streamlit app), checked 2026-11-08

| Item | Value |
|---|---|
| OS | Windows 11 Home (10.0.26300) |
| CPU | Intel Core i5-12450H, 8 cores / 12 threads |
| RAM | 7.6 GB (≈1–1.2 GB free during checks) |
| GPU | NVIDIA RTX 3050 Laptop, 4 GB VRAM, driver 617.42 (not used: PyTorch here is the CPU build) |
| Free disk | D: ≈30 GB |
| Python | 3.12.10 in `.venv` (created with uv 0.13.6) |

Key packages (full list: `requirements.lock.txt`): streamlit 1.67.0, torch 2.15.0+cpu, transformers 5.22.2, mediapipe 1.2.2, opencv 5.0.0, numpy 2.5.4, pandas 3.1.1, pillow 13.0.0, fastapi 0.162.2, pytest 9.2.0.

Notes:
- MediaPipe 1.x has removed the legacy `mediapipe.solutions` API; we use the Tasks API (`PoseLandmarker`).
- The try-on model does not run on this laptop (see `phase1_feasibility.md` §7).

## GPU server: Google Colab (free tier)

The notebook prints Python, torch, GPU name, memory and compute capability at the start of each session. Values from the first real session will be recorded here: **PENDING**.

## Sandbox used for compatibility testing

Linux, Python 3.12.3, torch 2.15.0, diffusers 0.42.1, transformers 5.22.2, CatVTON @ 7818397 (no GPU; random-weight smoke test only).
