"""About the project."""

import streamlit as st

from config import settings
from recommendation.knowledge import default_knowledge
from ui import state, theme

kb = default_knowledge()
theme.page_header("About the Project", "How Trycraz works, what is ours, what is borrowed, and where it falls short.")

st.markdown("""
**Problem.** Image-based virtual try-on shows how a garment looks on a person, but it does not say
whether the garment is *appropriate*: for an interview, a wedding, or a rainy day outdoors. This project
combines a pretrained try-on model with a context-aware, explainable recommender, so the system answers
both *"how does it look on me?"* and *"should I wear it there?"*.

**What is ours and what is borrowed.** The image synthesis is done by the pretrained **CatVTON** model;
we did not train or modify it. Our work is the system around it: input validation and pose-based quality
checks, garment background removal, the GPU server and client, the context model and recommender,
the catalogue, the integration and the evaluation.
""")

theme.section("Virtual try-on pipeline")
st.markdown("""
| Step | Technique | Implemented by |
|---|---|---|
| Input checks | Format/size/EXIF validation; brightness; sharpness (variance of Laplacian) | ours |
| Pose quality gate | MediaPipe BlazePose landmarks + geometric rules (visibility, frontal view, framing) | ours (pretrained pose model) |
| Garment preprocessing | U²-Net salient-object segmentation → garment on white, cropped | ours (pretrained U²-Net via rembg) |
| Body-part map | DensePose (Detectron2 R50-FPN) | pretrained, inside CatVTON |
| Human parsing | SCHP on LIP and ATR label sets | pretrained, inside CatVTON |
| Clothing-agnostic mask | Rule-based fusion of DensePose + SCHP; protects face, hands, hair, other garments | CatVTON code |
| Synthesis | Stable Diffusion 1.5 inpainting UNet; person and garment latents **concatenated side by side**; only self-attention was fine-tuned (49.6 M of 899 M parameters); no text encoder, no warping network | pretrained CatVTON |
| Output | Optional repaint of unmasked pixels from the original; comparison, download, saved runs | ours |

*Why concatenation works:* self-attention lets every pixel of the person half attend to every pixel of
the garment half, so the model learns where each part of the garment should go. This replaces the
explicit warping (TPS / flow) stage of older methods such as CP-VTON and VITON-HD.
""")

theme.section("Recommendation methodology")
weights = ", ".join(f"{k} {v:.2f}" for k, v in kb.weights.items())
st.markdown(f"""
Each garment gets six factor scores in [0, 1], combined as a weighted sum ({weights}):

- **Occasion**: catalogue tags, or style overlap for uploaded garments; garments conventionally avoided for the occasion score 0
- **Formality**: distance between the garment's formality (1–5) and the occasion's target, shifted by your preference
- **Weather**: warmth match for the weather, softened indoors, plus breathability (hot) or water resistance (rain, outdoors)
- **Colour / style**: your preferences, or the occasion's colour conventions if you set none
- **CLIP**: zero-shot classification of the garment **photo** over all {len(kb.occasions)} occasions and {len(kb.weather)} weather types (prompt ensembling)

Hard exclusions remove clearly inappropriate items (e.g. shorts for an interview) and are listed separately.
Explanations are generated from the same factor values as the ranking, so they cannot contradict it.
All dress-code knowledge is in `config/recommendation.yaml` and is our own conventional judgement, not learned from data.
""")

theme.section("Limitations")
st.markdown("""
- **Garment photos matter most.** CatVTON was trained on product photos on white. Home photos with cluttered
  backgrounds can produce wrong colours or textures; background removal reduces this but does not eliminate it.
- Works best on **front-facing** photos with arms at the sides; side views, crossed arms and occlusions cause artefacts.
- The demo catalogue images are **low resolution** (400 px), which limits fine detail.
- **No traditional Indian wear** in the catalogue yet (kurta, saree). Draped garments such as sarees are outside what this model supports.
- Dress-code knowledge is hand-written and culturally specific; recommendations are suggestions, not rules.
- Try-on needs a GPU: here, a free Google Colab session, which can disconnect.
""")

theme.section("Credits and licences")
st.markdown(f"""
- **CatVTON**: Chong et al., *Concatenation Is All You Need for Virtual Try-On with Diffusion Models*, ICLR 2025, CC BY-NC-SA 4.0
- **Stable Diffusion 1.5 inpainting**: CreativeML OpenRAIL-M · **DensePose** / **Detectron2** (Meta) · **SCHP** (Li et al.)
- **CLIP** ViT-B/32 (OpenAI, MIT) · **MediaPipe** Pose (Google, Apache 2.0) · **U²-Net** (Qin et al., Apache 2.0) via **rembg** (MIT)
- Catalogue images: **Clothing Dataset** by A. Grigorev et al. (CC0), see `data/catalogue/ATTRIBUTION.md`
- Sample people: Unsplash photographers Praise Judah, McFollis and Mohamad Khosravi (Unsplash License), see `assets/sample_people/ATTRIBUTION.md`

Trycraz is an academic, non-commercial project (B.Tech Computer Vision course, IIT Jodhpur).
""")

garments, _ = state.catalogue_or_error()
st.caption(f"Catalogue: {len(garments)} garments · CLIP model: {settings.CLIP_MODEL_NAME} · "
           f"try-on resolution {settings.MODEL_WIDTH}×{settings.MODEL_HEIGHT}")
theme.footer()
