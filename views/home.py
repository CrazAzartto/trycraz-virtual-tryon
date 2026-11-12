"""Home page."""

import streamlit as st

from ui import theme

theme.hero(
    theme.APP_NAME,
    theme.TAGLINE,
    "Trycraz dresses your photo in a garment with a pretrained diffusion model, and tells you whether that "
    "garment suits where you are going: an interview, a wedding, college, or a rainy day outdoors.",
)

left, right = st.columns(2, gap="large")
with left.container(border=True):
    theme.section("Virtual try-on")
    st.markdown(
        "Upload a photo of yourself and a garment, or pick one from the catalogue. The **CatVTON** diffusion "
        "model repaints only your clothing region and keeps your face, hair, body and background.\n\n"
        "- Tops, bottoms and full outfits\n"
        "- Photo check with pose estimation before generating\n"
        "- Automatic garment background removal\n"
        "- Drag-to-compare result and download"
    )
    if st.button("Start virtual try-on", type="primary", width="stretch"):
        st.switch_page("views/tryon.py")
with right.container(border=True):
    theme.section("Context-aware recommendations")
    st.markdown(
        "Tell Trycraz the occasion, the setting and the weather. It ranks the catalogue for menswear or "
        "womenswear and explains every recommendation.\n\n"
        "- Occasion, formality, weather, colour and style matching\n"
        "- CLIP image understanding as a learned signal\n"
        "- Check any garment of your own against the occasion\n"
        "- One click to try a recommendation on"
    )
    if st.button("Get recommendations", type="primary", width="stretch"):
        st.switch_page("views/recommend.py")

st.write("")
theme.section("How it works")
theme.steps([
    ("Choose the context", "Men or women, occasion, setting and weather. Colour and style preferences are optional."),
    ("Get ranked suggestions", "Every garment is scored on six factors; the explanation comes from the same scores."),
    ("Try it on", "Your photo and the garment go to a GPU server running CatVTON. About a minute later you see the result."),
    ("Compare and decide", "Slide between the original and the try-on, check the occasion verdict, try the next one."),
])

architecture = theme.UI_DIR / "architecture.png"
if architecture.exists():
    with st.expander("System architecture"):
        st.image(str(architecture), caption="The laptop runs the app and the recommender; a cloud GPU runs CatVTON.")
theme.footer()
