# Test Report

Only results that were actually executed are recorded here. Anything not yet run is marked **PENDING**.

## 1. Automated tests (pytest)

| Run | Environment | Command | Result |
|---|---|---|---|
| 2026-11-08 | Sandbox: Linux, Python 3.12.3, torch 2.15.0, streamlit 1.67.0, mediapipe 1.2.2 | `pytest -m "not slow"` | **63 passed**, 1 deselected (slow) |
| 2026-11-08 | Laptop: Windows 11, Python 3.12.10 (`.venv`), versions in `requirements.lock.txt` | `pytest` (including slow) | **64 passed** in 27 s (end of Phase 3) |
| 2026-11-08 | Sandbox (as above) | `pytest` | **156 passed** (after Phase 4 code) |
| 2026-11-08 | Laptop (as above) | `pytest` | **156 passed** in 19 s (after Phase 4 code) |
| 2026-11-08 | Sandbox | `pytest` (after Phase 5) | **164 passed** |
| 2026-11-08 | Laptop | `pytest` incl. slow (real MediaPipe, U²-Net) after Phase 5 | **164 passed** in 23 s |
| 2026-11-12 | Laptop | `pytest` | **164 passed, 3 failed**: Windows Smart App Control blocked the pandas/pyarrow DLLs (`An Application Control policy has blocked this file`), which broke `st.dataframe` |
| 2026-11-12 | Sandbox, then laptop | `pytest` after replacing `st.dataframe` with an HTML table and dropping pandas from the direct requirements | **166 passed** on both (laptop: 28 s) |

### 1.1 Real MediaPipe pose check on the laptop (CatVTON demo photos, CPU)

| Image | Size | People | Shoulder/torso ratio | Body height fraction | Issues (upper / lower / overall) |
|---|---|---|---|---|---|
| demo man (studio, full body) | 960×1280 | 1 | 0.64 | 0.75 | none / none / none |
| demo woman (studio) | 768×1024 | 1 | 0.65 | 0.96 | none / none / none |

The drawn skeleton was inspected visually: shoulders, elbows, wrists, hips, knees and ankles were all placed correctly.

### What the tests cover

| File | Covers |
|---|---|
| `test_environment.py` | every required package imports; settings load; 3:4 model resolution |
| `test_validation.py` | missing image, wrong extension, non-image bytes, truncated JPEG, unsupported real format (BMP), too small, extreme aspect ratio, file too large, RGBA→RGB on white, EXIF rotation, garment-type and parameter checks |
| `test_preprocessing.py` | centre-crop arithmetic identical to CatVTON's `resize_and_crop`; padding never crops the garment; no upscaling; pixel-aligned preview; mask overlay; comparison sheet |
| `test_tryon_server.py` | HTTP contract with the mock engine: health, API-key auth, job lifecycle and progress, parameter rejection (422), corrupt/empty uploads (400), GPU OOM → 507, generation failure → 500, unknown job → 404 |
| `test_remote_backend.py` | laptop client against a real uvicorn server on localhost: success with progress, wrong key, invalid request, unreachable server, not configured, OOM mapping |
| `test_quality_and_service.py` | pose rules on synthetic landmarks (frontal OK, missing hips/knees per garment type, side view, back view, hand on torso), darkness, blur, busy garment background; service never saves mock output to the gallery; [slow] real MediaPipe on a no-person image |
| `test_colab_notebook.py` | the notebook embeds the current server code byte-for-byte; every notebook cell compiles |
| `test_recommendations.py` | knowledge-base and catalogue validation (with row numbers); factor monotonicity; indoor/outdoor weather; scenario behaviour (interview, casual + hot, outdoor rain, cold, wedding, sports); hard exclusions; ranking changes with context; colour preference and formality shift; category / wear-line filters; determinism; **explanations consistent with factor values for all 9 occasions × 3 weathers**; no negative wording in reasons (9 × 4); CLIP weight redistribution; colour naming; uploaded-garment assessment |
| `test_rec_metrics.py` | Precision@k, nDCG@k, Kendall's tau against hand-computed values |

Scenario assertions were written from conventional dress-code expectations before reading the engine's output, and use a small **test fixture** catalogue (`tests/fixtures/catalogue_fixture.csv`, 18 metadata-only items). This is not the demo catalogue and not an evaluation; the real evaluation (P@5 / nDCG@5 with independent labels) runs in Phase 6 on the demo catalogue.

The mock engine is a test double that returns a 50/50 blend. It exists only to test plumbing and is labelled `MOCK (test only)` everywhere. No mock output is reported as a try-on result.

## 2. CatVTON compatibility smoke test (random weights)

`tests/compat/smoke_catvton_random_weights.py`: runs CatVTON's real code and our engine wrapper with **randomly initialised** tiny models on CPU. This checks that the code runs, not the image quality.

| Date | Stack | Steps exercised | Result |
|---|---|---|---|
| 2026-11-08 | Python 3.12.3, torch 2.15.0, diffusers 0.42.1, transformers 5.22.2, CatVTON @ 7818397 | detectron2 + DensePose R50-FPN build and inference; SCHP LIP + ATR; mask fusion (`AutoMasker`); `CatVTONPipeline` denoising loop with spatially concatenated latents and CFG; our `CatVTONEngine.generate` (progress counting, mask blur, repaint, timings) | **PASS**: mask outputs 768×1024; 5/5 progress events; pixels outside the mask identical to input after repaint |

Conclusion: CatVTON's vendored detectron2 binary (built for Python 3.9) is **not needed**; the Python fallbacks work on 3.12. This removes a Phase 1 risk.

## 2b. Real CLIP on the laptop (CPU), 2026-11-08

`openai/clip-vit-base-patch32` via transformers 5.22.2, on 8 CatVTON demo garment photos (retailer product images, used **only** for this check, not in the catalogue).

| Measure | Result |
|---|---|
| First load (including ~600 MB download) | 149 s |
| Per-garment analysis (embedding + type + colours) | 0.25–0.9 s |
| Garment-type top-1 plausible | 7 of 8 (3 dresses → `dress_casual` ×2 + `tank_top` for a one-shoulder draped mini dress; cardigan ✓; plain tee ✓; sleeveless knit → `tank_top`; football jersey ambiguous: top 0.30 / sweatshirt 0.25 / t-shirt 0.21) |
| Colour naming (k-means in CIELAB) | correct main colour for 6 of 8; failures: cream knit on white background (too close to background, ΔE < 18) and a multicolour floral print |

These are informal sanity checks on 8 images, not an accuracy measurement. Auto-annotations are therefore always reviewed by a person (`annotation=auto` → `verified`).

## 3. Defects found and fixed

| # | Found by | Defect | Fix |
|---|---|---|---|
| 1 | Compatibility smoke test | Our helper package `utils/` shadowed CatVTON's top-level `utils.py` (`ImportError: cannot import name 'compute_vae_encodings'`) whenever both were imported in one process | Renamed ours to `common/`; the engine now refuses to start if a foreign `utils`/`model` module is already loaded, with a clear message |
| 2 | CLI run on CatVTON demo photos | Blur check flagged sharp studio photos (Laplacian variance 50–68 against threshold 60) | Recalibrated on 20 demo images (sharp: 50–1562; blurred with σ = 1.5: 2–28), threshold set to 35 |
| 4 | Reading the engine's actual output | An explanation read "Recommended because … more casual (3.5/5) than the target 4.3": a factor scored ≥ 0.7 but its phrase was chosen with a different cut-off | Each factor now returns a positive and a negative phrase; the explainer picks one using the same thresholds as the scores. Regression test checks that no reason contains negative wording (36 contexts). Verdict now depends on the total score ("Recommended" / "A reasonable option" / "Weak match") |
| 3 | Code reading (PyTorch ≥ 2.6) | SCHP loader calls `torch.load` without `weights_only=False`; newer PyTorch defaults to `True` | `trusted_torch_load()` context manager, used only for the official checkpoints. **Effect on real weights: PENDING** (Colab) |

## 4. Real try-on (CatVTON on a Colab T4)

Server: CatVTON @ 7818397, `mix-48k-1024` attention weights, fp16, safety checker on, 768×1024, 30 DDIM steps, CFG 2.5, seed 42, repaint on. Model load: 86 s (warm Hugging Face cache on Colab).
Client: laptop → Cloudflare quick tunnel → job API. All runs below were executed on 2026-11-08 and inspected visually; "result" is our visual judgement, not a metric.

### 4.1 Performance (every run)

| Measure | Value over 10 runs |
|---|---|
| Diffusion time, 30 steps | 61.5–62.4 s (≈ 2.1 s/step) |
| Mask (DensePose + 2×SCHP) | 0.94–1.36 s |
| Peak GPU memory | 4.89 GB (of 15 GB on the T4) |
| Round trip from the laptop | 69–133 s (upload, tunnel, queue waiting) |

### 4.2 Results

| # | Person | Garment (photo background) | Type | Result | Notes |
|---|---|---|---|---|---|
| R1 | CatVTON demo man | demo black graphic tee (product photo) | upper | ✅ success | graphic and text preserved; face, shorts, background untouched; hands slightly smudged |
| R2 | man, photo cut off at hips | navy blazer (door + pink wall) | upper | ❌ failure | grey hooded top; halo around shoulders |
| R3 | man, photo cut off at hips | white formal shirt (wooden floor) | upper | ❌ failure | collar right, but grey knit-like texture; halo |
| R4 | woman, front | white wrap blouse (plain white wall) | upper | ✅ success | realistic drape and V-neck; hands and trousers untouched |
| R5 | woman, front | black formal trousers (red bedsheet) | lower | ⚠️ partial | trousers generated but red patches from the bedsheet and odd drape |
| R6 | man, full body, outdoors | light blue jeans (beige sheet) | lower | ✅ success | colour, fading and fit convincing; shirt and shoes untouched |
| R7 | woman in navy dress | tropical print dress (wooden floor) | overall | ⚠️ partial | print and colours transferred; silhouette changed to a sleeveless shift |

### 4.3 Diagnosing R2/R3 (hypothesis → test → evidence)

| Step | Hypothesis | Test | Evidence | Conclusion |
|---|---|---|---|---|
| 1 | Cluttered garment background confuses CatVTON | Remove background with U²-Net, rerun R2 | still grey + halo | **rejected** as the main cause of R2 |
| 2 | The garment itself is the problem | A: same segmented blazer on the demo man; B: on the woman | A ✅ navy blazer with lapels and button; B ✅ dark blazer | **rejected**: the garment works |
| 3 | The person photo is the problem | C: known-good tee (from R1) on the hips-cropped man | upper half ✅, lower half grey filler | **confirmed** |
| 4 | Why that photo? | MediaPipe hip position on the 3:4 crop for all photos | failing photos: hips at 0.89 and 0.91 of the height; working photos: 0.43–0.50 | the photo ends at the hips; the masked region touches the bottom edge and the model fills the part with no body below it with grey |

Actions: (a) new quality rule `cut_off_at_hips` (hip line below 85% of the cropped height is an error, below 75% a warning), now checked on the same 3:4 crop the model sees; (b) the photo was removed from the samples and kept in `assets/failure_examples/`; (c) background removal is kept anyway, because R5 shows background colour bleeding into the garment.

### 4.4 Validation after the fixes

New sample photos (full body, hips at about half the image height) and catalogue garments with the background removed (U²-Net).

| # | Person | Garment | Type | Result | Notes |
|---|---|---|---|---|---|
| V1 | man, studio, full body | navy blazer | upper | ✅ success | structured blazer, lapels, button; slightly lighter than the original; lining drawn as an inner layer |
| V2 | man, studio, full body | white formal shirt (failed as R3) | upper | ✅ success | collar, buttons, long sleeves; looks slightly sheer |
| V3 | woman, front | black formal trousers (partial as R5) | lower | ✅ success | no colour bleed |
| V4 | man in grey sweater | grey elephant-print T-shirt | upper | ✅ success | print preserved; sleeves shortened and forearms generated |
| V5 | woman in navy dress | royal blue sheath dress | overall | ✅ success | colour and shape right; sleeve length differs slightly |
| V6 | man, studio, in shorts | light blue jeans | lower | ✅ success | legs covered, fading convincing, shoes untouched |

6/6 successful after the two fixes, against 2 successes, 2 partials and 2 failures among R2–R7 before them. These are small numbers judged by eye. They show the fixes address the diagnosed causes, not a general success rate. Figures: `docs/figures/tryon_results.jpg`, `docs/figures/failure_diagnosis.jpg`.

### 4.5 Error handling against the real server

| Test | Expected | Actual | Status |
|---|---|---|---|
| Wrong API key from the laptop | clear message, no traceback | "The try-on server rejected the API key. Copy the key printed by the Colab notebook." (exit code 4) | ✅ PASS |
| Server URL that does not exist | clear message | "Cannot reach the try-on server. The Colab session may have stopped or the URL changed." (exit code 3) | ✅ PASS |
| Request queued while another job runs | waits, then completes | blazer job showed "queued" for ~45 s, then completed | ✅ PASS |

_Follow-up from §4.5: "Check connection" used to say *Connected* even with a wrong key, because `/health` needs no key. It now also probes an authenticated endpoint and reports a wrong key immediately (test added)._

## 5. Recommender evaluation (demo catalogue, real CLIP on the laptop)

Setup: 52-garment demo catalogue; 12 scenarios with graded relevance labels (`evaluation/rec_scenarios.yaml`) written **before** running the evaluation; k = 5. Command: `python -m evaluation.rec_eval --catalogue data/catalogue/clothing_catalogue.csv --scenarios evaluation/rec_scenarios.yaml --clip`.

| Configuration | mean P@5 | mean nDCG@5 |
|---|---|---|
| Attributes only (no CLIP) | 0.767 | 0.751 |
| CLIP only | 0.667 | 0.691 |
| **Hybrid (default weights)** | **0.833** | **0.830** |

Context sensitivity: mean Kendall τ over the 45 scenario pairs that share ≥ 5 garments = 0.07 (attributes only) and 0.01 (hybrid). τ = 1 would mean the ranking ignores the context.

Weakest scenarios (attributes only): *Party* (nDCG@5 0.28) and *College tops* (0.39; a rain jacket and a camisole reach the top 5 because their derived occasion tags are broad). The weights were **not** tuned on these labels: tuning on the evaluation set would inflate the numbers.

Caveats: one labeller, who also wrote the knowledge base and the annotations, so the numbers are likely optimistic; only 12 scenarios; a small catalogue. A second, independent labeller is planned.

Full per-scenario tables: `evaluation/results/rec_eval.md`.

## 6. Automatic annotation vs manual labels (52 garments)

| Measure | Before background removal | After background removal |
|---|---|---|
| Garment type: CLIP zero-shot top-1 = manual (26 classes) | 32/52 (62%) | 33/52 (63%) |
| Main colour: k-means name = manual main colour | 26/52 (50%) | 30/52 (58%) |

Removing the background mainly helps colour naming, because bedsheet and floor pixels no longer enter the colour clusters. Small sample, one annotator. Details: `evaluation/results/auto_annotation_agreement.md`.

## 7. UI tests

- `tests/test_ui.py` (Streamlit AppTest, headless): Home and About render; Recommendations form → results for *interview* contain no hoodie, shorts or tank top; Try-On page renders; a CLIP load failure does not crash the app.
- Browser run (Playwright, sandbox, mock try-on server): Home → Recommend → "Try this on" (garment and its type carried over) → Generate → result with comparison slider, MOCK warning, downloads and occasion verdict → About. 0 exceptions, 0 console errors.
- Two defects found by the browser run and fixed, each with a regression test: (5) the CLIP fallback wrote to a widget's own session key; (6) a form key collided with a session-state key ("context"). All app session keys now use the prefix `vto_`.
