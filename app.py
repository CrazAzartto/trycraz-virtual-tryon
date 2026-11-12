"""Trycraz: Streamlit entry point.

Run from the project root (or double-click start_app.bat):
    .venv\\Scripts\\streamlit run app.py

This file sets up the page config, theme, navigation and the shared sidebar. Each page lives in
views/, and all logic lives in virtual_tryon/ and recommendation/, so the UI stays thin.
"""

import streamlit as st

from ui import theme
from ui.components import sidebar

st.set_page_config(
    page_title=f"{theme.APP_NAME} | Virtual try-on and context-aware styling",
    page_icon=str(theme.FAVICON) if theme.FAVICON.exists() else None,
    layout="wide",
)
theme.apply()

home = st.Page("views/home.py", title="Home", default=True)
tryon = st.Page("views/tryon.py", title="Virtual Try-On")
recommend = st.Page("views/recommend.py", title="Recommendations")
about = st.Page("views/about.py", title="About the Project")

navigation = st.navigation([home, tryon, recommend, about])
sidebar()
navigation.run()
