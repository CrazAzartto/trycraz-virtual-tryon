"""Virtual try-on page."""

from __future__ import annotations

import io
import time
from pathlib import Path

import streamlit as st
from PIL import Image

from common.image_utils import overlay_mask, side_by_side
from common.validation import ImageValidationError, validate_image_bytes
from config import settings
from recommendation.garment_analyzer import analyse_garment
from recommendation.knowledge import default_knowledge
from ui import components, state, theme
from virtual_tryon import quality_check
from virtual_tryon.backends.base import TryOnError
from virtual_tryon.garment_segmentation import remove_background
from virtual_tryon.postprocessing import list_saved_runs
from virtual_tryon.service import TryOnService

kb = default_knowledge()
TYPE_LABELS = {"upper": "Upper body (tops, shirts, jackets)", "lower": "Lower body (trousers, shorts, skirts)",
               "overall": "Full outfit (dresses)"}
SHORT_TYPE = {"upper": "top", "lower": "bottom", "overall": "dress"}
EXT = [e.strip(".") for e in settings.ALLOWED_EXTENSIONS]
SAMPLE_PREFIX = {"menswear": "man_", "womenswear": "woman_"}

theme.page_header("Virtual Try-On", "Your photo plus a garment, rendered by the CatVTON diffusion model in about a minute.")


@st.cache_data(show_spinner=False, max_entries=32)
def person_quality(image_bytes: bytes, garment_type: str):
    image = validate_image_bytes(image_bytes, None, "person").image
    return quality_check.check_person_image(image, garment_type)


@st.cache_data(show_spinner="Removing the garment background...", max_entries=32)
def segmented(image_bytes: bytes) -> tuple[bytes, bool, str]:
    image = validate_image_bytes(image_bytes, None, "garment").image
    result = remove_background(image)
    return state.to_bytes(result.image, "PNG"), result.applied, result.message


def load(image_bytes: bytes, role: str) -> Image.Image:
    return validate_image_bytes(image_bytes, None, role).image


audience = components.audience_selector("tryon", required=False)

# ---------------------------------------------------------------------------
# 1. Person
# ---------------------------------------------------------------------------
left, right = st.columns(2, gap="large")
person_card, garment_card = left.container(border=True), right.container(border=True)
with person_card:
    theme.section("1  Your photo")
    source = st.radio("Person photo", ["Sample photo", "Upload my photo"], horizontal=True, label_visibility="collapsed")
    person_bytes = None
    if source == "Upload my photo":
        upload = st.file_uploader("Front-facing photo, head to at least mid-thigh visible", type=EXT, key="person_upload")
        if upload is not None:
            try:
                validate_image_bytes(upload.getvalue(), upload.name, "person")
                person_bytes = upload.getvalue()
            except ImageValidationError as exc:
                st.error(exc.user_message)
    else:
        samples = sorted(settings.SAMPLE_PEOPLE_DIR.glob("*.jpg"))
        if audience:
            samples = [p for p in samples if p.stem.startswith(SAMPLE_PREFIX[audience])] or samples
        if samples:
            choice = st.selectbox("Sample person", samples, format_func=lambda p: p.stem.replace("_", " ").capitalize())
            person_bytes = Path(choice).read_bytes()
            st.caption("Free Unsplash photos; credits on the About page.")
        else:
            st.warning("No sample photos found in assets/sample_people.")
    with st.expander("Tips for a good photo"):
        st.markdown("- Face the camera, arms relaxed at your sides\n- Head to at least mid-thigh visible "
                    "(whole body for trousers and dresses)\n- Plain background, even light, nobody else in the frame")

# ---------------------------------------------------------------------------
# 2. Garment
# ---------------------------------------------------------------------------
with garment_card:
    theme.section("2  Garment")
    garments, cat_error = state.catalogue_or_error()
    if audience:
        garments = [g for g in garments if g.wear_line in (audience, "unisex")]
    preselected = state.garment_by_id(st.session_state.get(state.TRYON_GARMENT_ID))
    options = ["From catalogue", "Upload a garment"] if garments else ["Upload a garment"]
    gsource = st.radio("Garment source", options, horizontal=True, label_visibility="collapsed")
    garment_bytes, garment_label, garment_id, suggested_type = None, "uploaded garment", None, "upper"
    if gsource == "From catalogue":
        ids = [g.id for g in garments]
        if preselected and preselected.id not in ids:
            ids.insert(0, preselected.id)
        default = ids.index(preselected.id) if preselected else 0
        names = {g.id: f"{g.name} ({SHORT_TYPE[g.tryon_type]})" for g in state.catalogue_or_error()[0]}
        gid = st.selectbox("Catalogue garment", ids, index=default, format_func=names.get)
        garment = state.garment_by_id(gid)
        garment_bytes, garment_label, garment_id, suggested_type = (garment.image_path.read_bytes(), garment.name,
                                                                    garment.id, garment.tryon_type)
        if preselected and preselected.id == gid:
            st.caption("Selected from your recommendations.")
    else:
        upload = st.file_uploader("Garment photo (product photo, laid flat or on a hanger)", type=EXT, key="garment_upload")
        if upload is not None:
            try:
                validate_image_bytes(upload.getvalue(), upload.name, "garment")
                garment_bytes, garment_label = upload.getvalue(), Path(upload.name).stem
            except ImageValidationError as exc:
                st.error(exc.user_message)
        if garment_bytes and st.toggle("Remove garment background (recommended)", value=True,
                                       help="CatVTON expects product photos on plain backgrounds. "
                                            "U²-Net segmentation puts the garment on white."):
            seg_bytes, applied, message = segmented(garment_bytes)
            if applied:
                garment_bytes = seg_bytes
            else:
                st.caption(message.capitalize())
        if garment_bytes and state.clip_or_none() is not None:
            analysis = analyse_garment(load(garment_bytes, "garment"), clip_scorer=state.clip_or_none())
            suggested_type = analysis.garment.tryon_type
            st.caption(f"Looks like: {kb.garment_types[analysis.garment.garment_type].label} (CLIP)")

    if garment_bytes:
        st.image(garment_bytes, width=220)
    garment_type = st.radio("Garment type", settings.GARMENT_TYPES, index=settings.GARMENT_TYPES.index(suggested_type),
                            format_func=TYPE_LABELS.get, help="Decides which body region the model replaces.")

# ---------------------------------------------------------------------------
# 3. Checks + generate
# ---------------------------------------------------------------------------
if person_bytes:
    with person_card:
        report = person_quality(person_bytes, garment_type)
        st.image(person_bytes, width=220)
        components.quality_report(report, "Photo")
        if report.landmarks is not None:
            with st.expander("Show detected pose"):
                st.image(quality_check.draw_skeleton(quality_check.model_view(load(person_bytes, "person")), report.landmarks),
                         width=300, caption="Detected pose on the part of the photo the model uses")
if garment_bytes:
    with garment_card:
        components.quality_report(quality_check.check_garment_image(load(garment_bytes, "garment")), "Garment")

with st.expander("Advanced settings"):
    a1, a2, a3, a4 = st.columns(4)
    steps = a1.slider("Denoising steps", 10, 60, settings.DEFAULT_STEPS, 5,
                      help="More steps: slightly more detail, slower (about 2 s per step on a T4).")
    guidance = a2.slider("Guidance (CFG)", 0.0, 5.0, settings.DEFAULT_GUIDANCE, 0.5, help="Higher values increase colour saturation.")
    seed = a3.number_input("Seed (-1 = random)", -1, 2**31 - 1, settings.DEFAULT_SEED)
    repaint = a4.toggle("Keep original pixels outside the garment", value=True)

ready = bool(person_bytes and garment_bytes)
if person_bytes and not person_quality(person_bytes, garment_type).ok:
    st.warning("The photo check found a problem that usually makes the try-on fail (see above). You can still try.")
if st.button("Generate try-on", type="primary", disabled=not ready, width="stretch"):
    progress = st.progress(0.0, text="Uploading...")
    labels = {"uploading": "Uploading", "queued": "Waiting for the GPU", "masking": "Finding the clothing region",
              "diffusion": "Generating", "done": "Finishing"}

    def on_progress(stage: str, step: int, total: int) -> None:
        fraction = 0.05 if stage in ("uploading", "queued", "masking") else min(1.0, 0.05 + 0.95 * step / max(total, 1))
        progress.progress(fraction, text=f"{labels.get(stage, stage)}... {f'step {step}/{total}' if stage == 'diffusion' else ''}")

    started = time.time()
    try:
        outcome = TryOnService(state.backend()).run(
            load(person_bytes, "person"), load(garment_bytes, "garment"), garment_type, int(steps), float(guidance),
            int(seed), repaint=repaint, garment_label=garment_label, garment_id=garment_id, progress=on_progress)
    except TryOnError as exc:
        progress.empty()
        st.error(exc.user_message)
        if exc.detail:
            st.caption(f"Details: {exc.detail}")
    else:
        progress.progress(1.0, text=f"Done in {time.time() - started:.0f} s")
        r = outcome.result
        st.session_state[state.LAST_OUTCOME] = state.StoredOutcome(
            person_png=state.to_bytes(outcome.prepared.person_preview), result_png=state.to_bytes(r.image),
            mask_png=state.to_bytes(r.mask), garment_png=state.to_bytes(outcome.prepared.garment),
            garment_label=garment_label, garment_id=garment_id, engine=r.engine, timings=r.timings,
            round_trip_s=r.round_trip_s, peak_gpu_memory_gb=r.peak_gpu_memory_gb, params=r.params,
            saved_folder=str(outcome.saved.folder) if outcome.saved else None)

# ---------------------------------------------------------------------------
# 4. Result
# ---------------------------------------------------------------------------
stored = st.session_state.get(state.LAST_OUTCOME)
if stored:
    st.divider()
    theme.section("Result")
    if "MOCK" in stored.engine:
        st.warning("This came from the MOCK test engine. It is not a try-on result.")
    person = Image.open(io.BytesIO(stored.person_png))
    result = Image.open(io.BytesIO(stored.result_png))
    mask = Image.open(io.BytesIO(stored.mask_png))
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        components.comparison_slider(person, result, width=420)
    with c2.container(border=True):
        st.markdown(f"**{stored.garment_label}**")
        st.caption(f"{TYPE_LABELS[stored.params.get('garment_type', 'upper')].split(' (')[0]} · {stored.engine} · "
                   f"{stored.params.get('steps')} steps · diffusion {stored.timings.get('diffusion_s', '?')} s · "
                   f"round trip {stored.round_trip_s} s · peak GPU {stored.peak_gpu_memory_gb} GB")
        ctx = st.session_state.get(state.CONTEXT)
        garment = state.garment_by_id(stored.garment_id)
        if ctx is not None and garment is not None:
            rec, exclusion = state.engine().assess(garment, ctx)
            st.markdown(f"**Suits {kb.occasions[ctx.occasion].label.lower()}?**")
            components.verdict_badge("Not suitable" if exclusion else rec.explanation.verdict)
            st.caption(exclusion or rec.explanation.summary)
        st.download_button("Download result (PNG)", stored.result_png, file_name="trycraz_result.png", mime="image/png",
                           width="stretch")
        sheet = side_by_side([person, Image.open(io.BytesIO(stored.garment_png)), overlay_mask(person, mask), result], height=512)
        st.download_button("Download comparison sheet (JPG)", state.to_bytes(sheet, "JPEG"), file_name="trycraz_comparison.jpg",
                           mime="image/jpeg", width="stretch")
        with st.expander("Region the model repainted"):
            st.image(overlay_mask(person, mask, color=(199, 154, 182)), width=260)
        if stored.saved_folder:
            st.caption(f"Saved to {stored.saved_folder}")

# ---------------------------------------------------------------------------
# Saved results (useful when the GPU server is offline)
# ---------------------------------------------------------------------------
runs = list_saved_runs()
if runs:
    with st.expander(f"Saved results from earlier sessions ({len(runs)})"):
        st.caption("Real CatVTON outputs saved on this computer, shown for reference (not generated now).")
        cols = st.columns(4)
        for i, run in enumerate(runs[:12]):
            with cols[i % 4]:
                st.image(str(run.folder / "result.png"), width="stretch")
                st.caption(f"{run.record.get('garment_label')} · {run.record.get('timestamp')}")
theme.footer()
