"""Context-aware recommendation page."""

from __future__ import annotations

import streamlit as st

from common.validation import ImageValidationError, validate_image_bytes
from config import settings
from recommendation.context_processor import Context
from recommendation.garment_analyzer import analyse_garment
from recommendation.knowledge import default_knowledge
from ui import components, state, theme

kb = default_knowledge()
CATEGORY_LABELS = {"upper": "Tops and jackets", "lower": "Bottoms", "overall": "Dresses"}

theme.page_header("Recommendations", "Tell us where you are going; we rank the catalogue and explain every pick.")
garments, error = state.catalogue_or_error()
if error:
    st.error(error)
    st.stop()

# ---------------------------------------------------------------------------
# 1. Who is it for (required)
# ---------------------------------------------------------------------------
audience = components.audience_selector("recommend", required=True)
if audience is None:
    st.session_state[state.LAST_RECOMMENDATIONS] = None
    theme.footer()
    st.stop()

# ---------------------------------------------------------------------------
# 2. Context form
# ---------------------------------------------------------------------------
with st.form("context_form"):
    c1, c2, c3 = st.columns([1.2, 1, 1.6])
    occasion = c1.selectbox("Occasion", list(kb.occasions), format_func=lambda k: kb.occasions[k].label)
    setting = c2.radio("Setting", list(kb.settings), format_func=lambda k: kb.settings[k], horizontal=True)
    weather = c3.radio("Weather", list(kb.weather), format_func=lambda k: kb.weather[k].label, horizontal=True)
    with st.expander("Optional preferences"):
        p1, p2 = st.columns(2)
        colours = p1.multiselect("Preferred colours", list(kb.colours), format_func=lambda c: c.replace("_", " "))
        styles = p2.multiselect("Preferred styles", list(kb.styles), format_func=lambda s: s.replace("_", " "))
        shift = p1.slider("More casual  /  More formal", -1.0, 1.0, 0.0, 0.25)
        default_categories = [c for c in settings.GARMENT_TYPES if not (audience == "menswear" and c == "overall")]
        categories = p2.multiselect("Garment categories", list(settings.GARMENT_TYPES), default=default_categories,
                                    format_func=CATEGORY_LABELS.get)
    top_k = st.slider("Number of recommendations", 3, 12, settings.DEFAULT_TOP_K)
    submitted = st.form_submit_button("Recommend", type="primary")

if submitted:
    if not categories:
        st.warning("Select at least one garment category.")
        st.stop()
    ctx = Context(occasion=occasion, setting=setting, weather=weather, colours=tuple(colours), styles=tuple(styles),
                  formality_shift=shift, wear_line=audience, categories=tuple(categories))
    with st.spinner("Ranking the catalogue..."):
        st.session_state[state.LAST_RECOMMENDATIONS] = state.engine().recommend(ctx, top_k=top_k)
    st.session_state[state.CONTEXT] = ctx

result = st.session_state.get(state.LAST_RECOMMENDATIONS)
if result is not None and result.profile.context.wear_line != audience:
    result = None   # the Men/Women choice changed since these results were computed
if result is None:
    theme.note("Choose the occasion, setting and weather, then press Recommend.")
    theme.footer()
    st.stop()

# ---------------------------------------------------------------------------
# 3. Results
# ---------------------------------------------------------------------------
profile = result.profile
who = "Men" if audience == "menswear" else "Women"
st.markdown(f"## {profile.occasion.label} &middot; {profile.weather.label} &middot; {profile.context.setting.capitalize()}")
st.caption(f"{who} · " + " · ".join(profile.notes) + (" · CLIP on" if result.clip_used else " · CLIP off"))

if not result.items:
    theme.note("No garment in the catalogue matches these filters. Try more categories or another occasion.")
cols = st.columns(3, gap="medium")
for i, rec in enumerate(result.items):
    g = rec.garment
    with cols[i % 3].container(border=True):
        if g.image_path and g.image_path.exists():
            st.image(str(g.image_path), width="stretch")
        st.markdown(f"**{rec.rank}. {g.name}**")
        components.verdict_badge(rec.explanation.verdict)
        components.score_bar(rec.score)
        st.caption(rec.explanation.summary)
        with st.expander("Why this score"):
            components.factor_table(rec, result.weights)
            st.caption(f"{kb.garment_types[g.garment_type].label} · formality {g.formality:g}/5 · warmth {g.warmth:g}/5 · "
                       f"{', '.join(c.replace('_', ' ') for c in g.colours)}")
        if st.button("Try this on", key=f"try_{g.id}", width="stretch"):
            st.session_state[state.TRYON_GARMENT_ID] = g.id
            st.switch_page("views/tryon.py")

shown_excluded = [(g, why) for g, why in result.excluded if "selected garment categories" not in why and "listed as" not in why]
if shown_excluded:
    with st.expander(f"Not suitable for this occasion ({len(shown_excluded)})"):
        for g, why in shown_excluded:
            st.caption(f"{g.name}: {why}")

# ---------------------------------------------------------------------------
# 4. Check your own garment
# ---------------------------------------------------------------------------
st.divider()
theme.section("Is my own garment suitable?")
upload = st.file_uploader("Upload a garment photo", type=[e.strip(".") for e in settings.ALLOWED_EXTENSIONS], key="own_garment")
if upload is not None:
    try:
        image = validate_image_bytes(upload.getvalue(), upload.name, "garment").image
    except ImageValidationError as exc:
        st.error(exc.user_message)
        st.stop()
    scorer = state.clip_or_none()
    a, b = st.columns([1, 2], gap="large")
    a.image(image, width="stretch")
    with b:
        if scorer is None:
            gtype = st.selectbox("What is it?", list(kb.garment_types), format_func=lambda t: kb.garment_types[t].label)
            analysis = analyse_garment(image, garment_type=gtype)
        else:
            with st.spinner("Analysing the garment..."):
                analysis = analyse_garment(image, clip_scorer=scorer)
            guess = ", ".join(f"{kb.garment_types[t].label} {p:.0%}" for t, p in analysis.type_candidates)
            st.caption(f"Detected by CLIP: {guess}")
            override = st.selectbox("Garment type", list(kb.garment_types),
                                    index=list(kb.garment_types).index(analysis.garment.garment_type),
                                    format_func=lambda t: kb.garment_types[t].label)
            if override != analysis.garment.garment_type:
                analysis = analyse_garment(image, garment_type=override)
        st.caption("Colours: " + ", ".join(f"{n.replace('_', ' ')} {s:.0%}" for n, s in analysis.colour.colours))
        clip_probs = scorer.probs_from_embedding(scorer.embed(image)) if scorer else None
        rec, exclusion = state.engine().assess(analysis.garment, profile.context, clip_probs)
        components.verdict_badge("Not suitable" if exclusion else rec.explanation.verdict)
        if exclusion:
            st.write(f"Not suitable: {exclusion}.")
        components.score_bar(rec.score)
        st.write(rec.explanation.summary)
        st.caption("Attributes were estimated automatically from the photo, so treat this as a guide.")
theme.footer()
