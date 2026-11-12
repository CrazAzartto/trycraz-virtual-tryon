"""Reusable Streamlit widgets (no emojis; colours from ui.theme)."""

from __future__ import annotations

import base64
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from virtual_tryon.quality_check import QualityReport

from . import state, theme
from .state import to_bytes

LEVEL_LABEL = {"error": "Problem", "warning": "Warning", "info": "Note"}
AUDIENCES = {"Men": "menswear", "Women": "womenswear"}


def sidebar() -> None:
    """Server connection + global options, shown on every page."""
    state.init_session()
    with st.sidebar:
        st.markdown('<div class="tc-section">Try-on server</div>', unsafe_allow_html=True)
        st.text_input("Server URL", key=state.SERVER_URL, placeholder="https://xxxx.trycloudflare.com",
                      help="Printed by the Colab/Kaggle notebook (step 4). It changes every time the notebook restarts.")
        st.text_input("API key", key=state.API_KEY, type="password", help="Printed by the notebook (step 4).")
        if st.button("Check connection", width="stretch"):
            with st.spinner("Contacting server..."):
                st.session_state[state.SERVER_STATUS] = state.backend().health()
        status = st.session_state.get(state.SERVER_STATUS)
        if status is not None:
            if status.reachable and status.is_mock:
                st.warning("Connected to the MOCK test engine: results are not real try-ons.")
            elif status.reachable:
                st.success(status.message)
            else:
                st.error(status.message)
        st.divider()
        st.markdown('<div class="tc-section">Recommendations</div>', unsafe_allow_html=True)
        st.toggle("Use CLIP image understanding", key=state.USE_CLIP,
                  help="Adds a learned visual signal to the ranking and recognises uploaded garments. "
                       "Loading takes 1-2 minutes the first time.")
        if st.session_state.get(state.CLIP_ERROR):
            st.caption(f"CLIP unavailable: {st.session_state[state.CLIP_ERROR]}")
            if st.button("Retry loading CLIP", width="stretch"):
                st.session_state[state.CLIP_ERROR] = None
                st.rerun()


def audience_selector(page_key: str, required: bool) -> Optional[str]:
    """"Shopping for: Men / Women". Returns 'menswear', 'womenswear' or None.

    The choice is shared by all pages through state.AUDIENCE. Each page uses its own widget key
    (Streamlit forgets a widget's state when another page does not draw it) and starts from the stored value.
    """
    stored = st.session_state.get(state.AUDIENCE)
    default = next((label for label, value in AUDIENCES.items() if value == stored), None)
    choice = st.segmented_control(
        "Shopping for", list(AUDIENCES), default=default, key=f"audience_{page_key}",
        help="Shows menswear or womenswear plus unisex garments.",
    )
    st.session_state[state.AUDIENCE] = AUDIENCES.get(choice) if choice else None
    if required and not choice:
        theme.note("Choose Men or Women to see recommendations.")
    return st.session_state[state.AUDIENCE]


def comparison_slider(before: Image.Image, after: Image.Image, width: int = 420) -> None:
    """Drag to compare the original photo (left) with the try-on result (right)."""
    b64 = lambda im: base64.b64encode(to_bytes(im.convert("RGB"), "JPEG", 90)).decode()
    height = int(width * after.height / after.width)
    label = (f"position:absolute;top:10px;background:rgba(23,11,19,.78);color:{theme.BLUSH_100};padding:3px 9px;"
             f"border-radius:999px;font:600 11px/1.4 Inter,Segoe UI,sans-serif;letter-spacing:.6px;"
             f"border:1px solid {theme.PLUM_500}")
    html = f"""
    <style>
      input[type=range]{{-webkit-appearance:none;appearance:none;height:4px;border-radius:4px;background:{theme.PLUM_500}}}
      input[type=range]::-webkit-slider-thumb{{-webkit-appearance:none;width:18px;height:18px;border-radius:50%;
        background:{theme.MAUVE_200};border:2px solid {theme.PLUM_700};cursor:ew-resize}}
      input[type=range]::-moz-range-thumb{{width:18px;height:18px;border-radius:50%;background:{theme.MAUVE_200};
        border:2px solid {theme.PLUM_700};cursor:ew-resize}}
    </style>
    <div style="position:relative;width:{width}px;height:{height}px;user-select:none;border-radius:12px;overflow:hidden;
                border:1px solid {theme.PLUM_600};box-shadow:0 12px 30px rgba(10,3,8,.55)">
      <img src="data:image/jpeg;base64,{b64(after)}" style="position:absolute;inset:0;width:100%;height:100%">
      <img id="before" src="data:image/jpeg;base64,{b64(before.resize(after.size))}"
           style="position:absolute;inset:0;width:100%;height:100%;clip-path:inset(0 50% 0 0)">
      <div id="bar" style="position:absolute;top:0;bottom:0;left:50%;width:2px;background:{theme.MAUVE_200};
           box-shadow:0 0 6px rgba(23,11,19,.8)"></div>
      <span style="{label};left:10px">ORIGINAL</span>
      <span style="{label};right:10px">TRY-ON</span>
      <input type="range" min="0" max="100" value="50" aria-label="Comparison position"
             style="position:absolute;bottom:14px;left:6%;width:88%"
             oninput="document.getElementById('before').style.clipPath='inset(0 '+(100-this.value)+'% 0 0)';
                      document.getElementById('bar').style.left=this.value+'%'">
    </div>"""
    components.html(html, height=height + 6, width=width + 6)


def quality_report(report: QualityReport, title: str) -> None:
    if not report.issues:
        st.markdown(f'<div class="tc-check"><b>{title} check</b> &middot; no problems detected</div>',
                    unsafe_allow_html=True)
        return
    for issue in report.issues:
        css = "tc-check tc-error" if issue.level == "error" else "tc-check"
        st.markdown(f'<div class="{css}"><b>{LEVEL_LABEL.get(issue.level, "Note")}</b> &middot; '
                    f'{theme.esc(issue.message)}</div>', unsafe_allow_html=True)


def score_bar(score: float) -> None:
    st.progress(min(max(score, 0.0), 1.0), text=f"Suitability {score:.2f}")


def verdict_badge(verdict: str) -> None:
    theme.verdict(verdict)


def factor_table(rec, weights: dict) -> None:
    """Six-factor breakdown as a small HTML table.

    Deliberately not st.dataframe: that needs pandas/pyarrow DLLs, which Windows Smart App Control
    may block (seen on the project laptop on 2026-11-12). A plain table needs neither."""
    rows = []
    for name, factor in rec.factors.items():
        if weights.get(name, 0) == 0:
            continue
        used = "yes" if factor.informative else "neutral"
        rows.append(f"<tr><td>{theme.esc(name.capitalize())}</td><td>{factor.value:.2f}</td><td>{weights[name]:.2f}</td>"
                    f"<td>{weights[name] * factor.value:.3f}</td><td>{used}</td></tr>")
    st.markdown('<table class="tc-table"><thead><tr><th>Factor</th><th>Value</th><th>Weight</th><th>Contribution</th>'
                f'<th>Used</th></tr></thead><tbody>{"".join(rows)}</tbody></table>', unsafe_allow_html=True)
