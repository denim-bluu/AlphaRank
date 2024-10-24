import streamlit as st

st.set_page_config(
    page_title="Performance Analysis Tool", page_icon="📊", layout="wide"
)


config_page = st.Page(
    "frontend/pages/config.py", title="Config", icon=":material/login:"
)
analysis_page = st.Page(
    "frontend/pages/analysis.py", title="Analysis", icon="🔬"
)

pg = st.navigation(
    {
        "Config": [config_page],
        "Analysis": [analysis_page],
    }
)

pg.run()
