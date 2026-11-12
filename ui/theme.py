"""Trycraz visual identity: berry palette, typography and a few HTML building blocks.

Colours come only from the berry/plum reference image. Base colours live in
.streamlit/config.toml (Streamlit's own theme); this file adds what the theme cannot express:
card styling, the hero banner, pills, step cards and small typographic touches.
"""

from __future__ import annotations

import html

import streamlit as st

from config import settings

APP_NAME = "Trycraz"
TAGLINE = "See it on you. Know where to wear it."

# Palette (also documented in .streamlit/config.toml)
PLUM_950 = "#170B13"
PLUM_900 = "#1F0E19"
PLUM_850 = "#24111E"
PLUM_800 = "#2E1627"
PLUM_700 = "#3A1B30"
PLUM_600 = "#4A2440"
PLUM_500 = "#72395D"
ROSE_400 = "#A86C90"
MAUVE_300 = "#C79AB6"
MAUVE_200 = "#D7AFC6"
BLUSH_100 = "#F2E6EE"
MUTED = "#BFA3B5"

UI_DIR = settings.ASSETS_DIR / "ui"
WORDMARK = UI_DIR / "wordmark.svg"
ICON = UI_DIR / "icon.svg"
FAVICON = UI_DIR / "favicon.png"

_CSS = f"""
<style>
/* ---------- layout ---------- */
.block-container {{ padding-top: 3.6rem; max-width: 1240px; }}
h1, h2, h3 {{ letter-spacing: 0.2px; }}
h1 {{ color: {BLUSH_100}; }}
h2, h3 {{ color: {MAUVE_200}; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: {MUTED} !important; }}
hr {{ border-color: {PLUM_700} !important; }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{ border-right: 1px solid {PLUM_700}; }}
[data-testid="stSidebarNav"] a span {{ font-weight: 500; letter-spacing: 0.2px; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: {PLUM_800}; border-left: 3px solid {ROSE_400}; }}
[data-testid="stLogo"] {{ height: 2.6rem; max-width: 14rem; }}

/* ---------- bordered containers become cards ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: linear-gradient(180deg, {PLUM_850} 0%, {PLUM_900} 100%);
    border-color: {PLUM_600} !important;
    box-shadow: 0 10px 28px rgba(10, 3, 8, 0.45);
}}
[data-testid="stVerticalBlockBorderWrapper"] img {{ border-radius: 0.5rem; }}

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
    font-weight: 600; letter-spacing: 0.3px; transition: all .15s ease;
}}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"] {{
    background: linear-gradient(135deg, {ROSE_400} 0%, {PLUM_500} 100%);
    border: 1px solid {MAUVE_300}; color: {BLUSH_100};
}}
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {{
    filter: brightness(1.08); border-color: {MAUVE_200};
}}
.stButton > button[kind="secondary"]:hover, .stDownloadButton > button:hover {{ border-color: {ROSE_400}; color: {MAUVE_200}; }}

/* ---------- segmented control (Men / Women): filled when selected ---------- */
[data-testid="stButtonGroup"] button {{ min-width: 6.5rem; font-weight: 600; letter-spacing: .3px; }}
[data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[data-selected="true"] {{
    background: linear-gradient(135deg, {ROSE_400} 0%, {PLUM_500} 100%) !important;
    color: {BLUSH_100} !important; border-color: {MAUVE_300} !important;
}}

/* ---------- inputs ---------- */
[data-testid="stExpander"] details {{ border-color: {PLUM_600} !important; background: {PLUM_900}; }}
[data-testid="stExpander"] summary:hover {{ color: {MAUVE_200}; }}
[data-testid="stProgress"] > div > div > div > div {{ background: linear-gradient(90deg, {PLUM_500}, {MAUVE_300}); }}

/* ---------- Trycraz components ---------- */
.tc-hero {{
    position: relative; overflow: hidden; border-radius: 1.1rem; padding: 2.6rem 2.6rem 2.2rem;
    background:
      radial-gradient(circle at 88% 18%, rgba(199,154,182,0.22) 0, rgba(199,154,182,0) 38%),
      radial-gradient(circle at 12% 110%, rgba(168,108,144,0.28) 0, rgba(168,108,144,0) 45%),
      linear-gradient(135deg, {PLUM_700} 0%, {PLUM_850} 55%, {PLUM_950} 100%);
    border: 1px solid {PLUM_600}; box-shadow: 0 18px 40px rgba(10,3,8,.55); margin-bottom: 1.6rem;
}}
.tc-eyebrow {{ text-transform: uppercase; letter-spacing: 3px; font-size: .74rem; color: {MAUVE_300}; font-weight: 600; }}
.tc-hero h1 {{ font-size: 3.4rem; margin: .35rem 0 .2rem; line-height: 1.05; color: {BLUSH_100}; padding: 0; }}
.tc-hero .tc-tagline {{ font-family: 'Playfair Display', Georgia, serif; font-style: italic; font-size: 1.35rem; color: {MAUVE_200}; }}
.tc-hero p {{ color: {MUTED}; max-width: 46rem; font-size: 1.02rem; line-height: 1.6; margin-top: .9rem; }}
.tc-page-head {{ margin: .2rem 0 1.4rem; padding-bottom: .9rem; border-bottom: 1px solid {PLUM_700}; }}
.tc-page-head h1 {{ margin: .2rem 0 .3rem; padding: 0; font-size: 2.4rem; }}
.tc-page-head p {{ color: {MUTED}; margin: 0; }}
.tc-section {{ font-family: 'Playfair Display', Georgia, serif; font-size: 1.35rem; color: {MAUVE_200}; margin: .2rem 0 .6rem; }}
.tc-pill {{
    display: inline-block; padding: .18rem .7rem; border-radius: 999px; font-size: .78rem; font-weight: 600;
    letter-spacing: .4px; border: 1px solid {PLUM_500}; background: {PLUM_800}; color: {MAUVE_200};
}}
.tc-pill.tc-strong {{ background: {ROSE_400}; border-color: {MAUVE_300}; color: {PLUM_950}; }}
.tc-pill.tc-mid {{ background: {PLUM_600}; border-color: {ROSE_400}; color: {BLUSH_100}; }}
.tc-pill.tc-weak {{ background: transparent; border-color: {PLUM_500}; color: {MUTED}; }}
.tc-pill.tc-no {{ background: {PLUM_950}; border-color: {PLUM_500}; color: {MUTED}; text-decoration: line-through; }}
.tc-steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: .4rem 0 1.2rem; }}
.tc-step {{ background: {PLUM_850}; border: 1px solid {PLUM_700}; border-radius: .9rem; padding: 1.1rem 1.2rem; }}
.tc-step .tc-num {{
    display: inline-flex; align-items: center; justify-content: center; width: 2rem; height: 2rem; border-radius: 50%;
    background: {PLUM_600}; color: {BLUSH_100}; font-family: 'Playfair Display', Georgia, serif; font-weight: 700;
    border: 1px solid {ROSE_400}; margin-bottom: .55rem;
}}
.tc-step h4 {{ margin: 0 0 .3rem; color: {BLUSH_100}; font-size: 1.02rem; }}
.tc-step p {{ margin: 0; color: {MUTED}; font-size: .92rem; line-height: 1.5; }}
.tc-note {{ border-left: 3px solid {ROSE_400}; background: {PLUM_850}; padding: .7rem 1rem; border-radius: .4rem;
            color: {MUTED}; font-size: .93rem; margin: .4rem 0 .8rem; }}
.tc-check {{ font-size: .88rem; color: {MUTED}; margin: .15rem 0; }}
.tc-check b {{ color: {MAUVE_200}; font-weight: 600; letter-spacing: .3px; }}
.tc-check.tc-error b {{ color: {BLUSH_100}; background: {PLUM_600}; padding: 0 .4rem; border-radius: .3rem; }}
.tc-table {{ width: 100%; border-collapse: collapse; font-size: .82rem; margin: .2rem 0 .5rem; }}
.tc-table th {{ text-align: left; color: {MAUVE_200}; font-weight: 600; border-bottom: 1px solid {PLUM_600}; padding: .3rem .4rem; }}
.tc-table td {{ color: {BLUSH_100}; border-bottom: 1px solid {PLUM_700}; padding: .3rem .4rem; }}
.tc-table td:nth-child(n+2):nth-child(-n+4) {{ font-variant-numeric: tabular-nums; }}
.tc-footer {{ color: {PLUM_500}; font-size: .8rem; text-align: center; margin-top: 2.5rem; }}
</style>
"""


def apply() -> None:
    """Inject the stylesheet and brand; call once per run (from app.py)."""
    st.markdown(_CSS, unsafe_allow_html=True)
    if WORDMARK.exists():
        st.logo(str(WORDMARK), icon_image=str(ICON) if ICON.exists() else None, size="large")


def esc(text: str) -> str:
    return html.escape(str(text))


def page_header(title: str, subtitle: str = "", eyebrow: str = APP_NAME) -> None:
    st.markdown(
        f'<div class="tc-page-head"><div class="tc-eyebrow">{esc(eyebrow)}</div><h1>{esc(title)}</h1>'
        f'{f"<p>{esc(subtitle)}</p>" if subtitle else ""}</div>',
        unsafe_allow_html=True,
    )


def hero(title: str, tagline: str, body: str) -> None:
    st.markdown(
        f'<div class="tc-hero"><div class="tc-eyebrow">Virtual try-on &middot; Context-aware styling</div>'
        f'<h1>{esc(title)}</h1><div class="tc-tagline">{esc(tagline)}</div><p>{esc(body)}</p></div>',
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f'<div class="tc-section">{esc(title)}</div>', unsafe_allow_html=True)


def steps(items: list[tuple[str, str]]) -> None:
    cards = "".join(f'<div class="tc-step"><div class="tc-num">{i}</div><h4>{esc(t)}</h4><p>{esc(d)}</p></div>'
                    for i, (t, d) in enumerate(items, start=1))
    st.markdown(f'<div class="tc-steps">{cards}</div>', unsafe_allow_html=True)


VERDICT_TONE = {"Recommended": "tc-strong", "A reasonable option": "tc-mid", "Weak match": "tc-weak", "Not suitable": "tc-no"}


def pill(text: str, tone: str = "") -> str:
    return f'<span class="tc-pill {tone}">{esc(text)}</span>'


def verdict(text: str) -> None:
    st.markdown(pill(text, VERDICT_TONE.get(text, "")), unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="tc-note">{esc(text)}</div>', unsafe_allow_html=True)


def footer() -> None:
    st.markdown(f'<div class="tc-footer">{APP_NAME} &middot; B.Tech Computer Vision project, IIT Jodhpur &middot; '
                f'try-on model: CatVTON (CC BY-NC-SA 4.0)</div>', unsafe_allow_html=True)
