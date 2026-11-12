# Progress Tracker

Every file created or modified, and the true status of every feature. Statuses: **Done (tested)**, **Done (untested)**, **In progress**, **Pending**.

## Phases

| Phase | Status | Notes |
|---|---|---|
| 1: Research & feasibility | **Done** (2026-11-08) | CatVTON selected; Leffa optional. See `phase1_feasibility.md` |
| 2: Project setup | **Done** (2026-11-08) | `.venv` on Python 3.12.10 via uv; git repository |
| 3: Virtual try-on | **Done** (2026-11-08) | Real CatVTON runs on a Colab T4 from the laptop; failure diagnosed (photo cut off at hips) and turned into a quality rule; garment background removal added |
| 4: Recommendation engine | **Done** (2026-11-08) | 52-garment demo catalogue (CC0 images, manual annotations); hybrid scoring; first evaluation run |
| 5: Integration / UI | **Done (first version)** (2026-11-08) | Four-page Streamlit app tested headlessly (AppTest) and in a browser with the mock server. Also run on the laptop against the Kaggle/Colab GPU |
| 6: Testing & evaluation | **In progress** | Recommender metrics + ablation done (one labeller). Pending: paired VITON-HD try-on metrics (SSIM/LPIPS/KID) on Colab, human rating sheet, second labeller |
| 7: Report, slides, viva | **Report done** (2026-11-12) | `docs/report/Trycraz_Project_Report.pdf` (IEEE format) with LaTeX source |

## Decisions (2026-11-08)
- GPU: free Google Colab only. Kaggle as backup.
- For now, use free-licensed images instead of our own: garments from the Clothing Dataset (CC0), people from Unsplash (Unsplash License). We plan to switch to our own photos later.
- Catalogue rows annotated by hand by the AI assistant are marked `annotation=manual` until the team confirms them (`verified`).

## Features

| Feature | Status |
|---|---|
| Input validation (missing/corrupt/format/size/aspect/EXIF) | Done (tested) |
| Client preprocessing (CatVTON-identical crop, pad, downscale) | Done (tested) |
| Quality gate: brightness, blur, garment background | Done (tested; blur threshold calibrated on 20 images) |
| Quality gate: MediaPipe pose rules incl. `cut_off_at_hips` | Done (tested; hip rule calibrated on real failures) |
| Garment background removal (U²-Net via rembg) | Done (tested; used in catalogue build and on uploads) |
| Try-on server (FastAPI job API, progress, API key) | Done (tested with mock + real CatVTON) |
| CatVTON engine wrapper | Done (real runs on T4: ~62 s / 30 steps, 4.9 GB peak) |
| Colab notebook | Done (run on Colab and Kaggle) |
| Remote / local backends, CLI | Done (tested) |
| Try-on: upper / lower / overall on real photos | Tested on real runs; see test report §4 for successes and failures |
| Knowledge base, scoring, exclusions, explanations | Done (tested) |
| CLIP factor + garment-type detection | Done (real CLIP on laptop) |
| Dominant-colour analysis | Done (58% main-colour agreement with manual labels) |
| Uploaded-garment suitability | Done (tested; in the UI) |
| Demo catalogue (52 garments) | Done; annotations `manual`, to be confirmed by the team |
| Recommender evaluation (P@5, nDCG@5, ablation, Kendall tau) | Done (one labeller) |
| Streamlit app: Home, Try-On, Recommendations, About | Done (AppTest + browser test with mock server) |
| Paired try-on metrics on VITON-HD | Pending |

## File log

| Date | File | Change |
|---|---|---|
| 2026-11-08 | `docs/phase1_feasibility.md`, `docs/architecture_phase1.png` | Created: Phase 1 research and architecture |
| 2026-11-08 | `README.md`, `LICENSE`, `.gitignore`, `pyproject.toml`, `requirements*.txt` | Created; requirements later gained rembg + onnxruntime |
| 2026-11-08 | `config/settings.py`, `config/recommendation.yaml` | Created |
| 2026-11-08 | `common/image_utils.py`, `common/validation.py` | Created (package renamed from `utils/`: name clash with CatVTON) |
| 2026-11-08 | `tryon_server/*` | Created: GPU-side server |
| 2026-11-08 | `virtual_tryon/*` | Created; `garment_segmentation.py` added after the first real runs; `quality_check.py` gained the hip-framing rule and runs on the model's 3:4 crop |
| 2026-11-08 | `recommendation/*` | Created |
| 2026-11-08 | `ui/state.py`, `ui/components.py`, `app.py`, `pages/*.py` | Created: Streamlit app |
| 2026-11-08 | `ui/theme.py`, `.streamlit/config.toml`, `assets/ui/*`, `views/*.py` | App renamed **Trycraz**; emojis removed; berry/plum theme; Men/Women choice required before recommendations (also filters the try-on page); `pages/` renamed `views/` so direct page URLs keep the themed navigation |
| 2026-11-08 | `scripts/build_colab_notebook.py`, `notebooks/colab_tryon_server.ipynb` | Created (notebook is generated) |
| 2026-11-08 | `scripts/tryon_cli.py`, `scripts/build_catalogue.py`, `scripts/apply_annotations.py`, `scripts/check_env.py` | Created |
| 2026-11-08 | `data/catalogue/*` | Created: 52-garment catalogue, manual annotations, attribution |
| 2026-11-08 | `assets/sample_people/*`, `assets/failure_examples/*` | Created: Unsplash photos + attribution |
| 2026-11-08 | `evaluation/rec_eval.py`, `evaluation/rec_scenarios.yaml`, `evaluation/results/*` | Created |
| 2026-11-08 | `tests/*` | Test suite (see test report for counts) |
| 2026-11-08 | `docs/test_report.md`, `docs/environment.md`, `docs/PROGRESS.md` | Created / updated |
| 2026-11-12 | `README.md`, `docs/SETUP.md`, `docs/ARCHITECTURE.md`, `docs/screenshots/`, `docs/report/` | Documentation for the public GitHub repository; project report |
