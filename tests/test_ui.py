"""Headless UI tests with Streamlit's AppTest (no browser).

Regression tests for two bugs found in a browser run on 2026-11-08:
  - writing to a widget's session key after the widget was drawn (CLIP fallback),
  - the form key "context" colliding with a session-state key of the same name.
CLIP is switched off so no model is downloaded.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from config import settings
from ui import state

ROOT = Path(__file__).resolve().parent.parent
needs_catalogue = pytest.mark.skipif(not settings.CATALOGUE_CSV.exists(), reason="demo catalogue not built")


def page(name: str) -> AppTest:
    at = AppTest.from_file(str(ROOT / "views" / name), default_timeout=60)
    at.session_state[state.USE_CLIP] = False
    return at


def test_home_and_about_render():
    for name, heading in (("home.py", "Trycraz"), ("about.py", "About the Project")):
        at = page(name).run()
        assert not at.exception, at.exception
        assert any(heading in m.value for m in at.markdown)   # headings are styled HTML, not st.title


@needs_catalogue
def test_recommend_requires_men_or_women_first():
    at = page("recommend.py").run()
    assert not at.exception
    assert not at.selectbox                # the context form is not shown until Men/Women is chosen
    assert at.session_state[state.LAST_RECOMMENDATIONS] is None


@needs_catalogue
@pytest.mark.parametrize("audience,forbidden", [("menswear", "womenswear"), ("womenswear", "menswear")])
def test_recommend_flow(audience, forbidden):
    at = page("recommend.py")
    at.session_state[state.AUDIENCE] = audience
    at.run()
    assert not at.exception, at.exception
    at.selectbox[0].set_value("interview")
    at.button[0].click().run()            # the form's "Recommend" submit button
    assert not at.exception, at.exception
    result = at.session_state[state.LAST_RECOMMENDATIONS]
    assert result.items and result.profile.occasion.key == "interview"
    assert all(rec.garment.garment_type not in ("hoodie", "shorts", "tank_top") for rec in result.all_ranked)
    assert all(rec.garment.wear_line != forbidden for rec in result.all_ranked)


def test_no_emojis_in_ui_code():
    import re
    emoji = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]")
    files = [ROOT / "app.py", *sorted((ROOT / "views").glob("*.py")), *sorted((ROOT / "ui").glob("*.py"))]
    offenders = [f"{f.name}:{i}" for f in files for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1)
                 if emoji.search(line)]
    assert not offenders, offenders


@needs_catalogue
def test_tryon_page_without_server_shows_guidance():
    at = page("tryon.py").run()
    assert not at.exception, at.exception
    assert any(b.label == "Generate try-on" for b in at.button)


def test_clip_failure_does_not_crash(monkeypatch):
    """CLIP load failure must not try to modify the toggle's own session key."""
    def boom():
        raise OSError("no internet")
    monkeypatch.setattr(state, "clip_scorer", boom)
    at = AppTest.from_string(
        "from ui import state\nstate.init_session()\nimport streamlit as st\n"
        "st.toggle('clip', key=state.USE_CLIP)\nst.write(state.clip_or_none())"
    )
    at.run()
    assert not at.exception, at.exception
