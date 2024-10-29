import streamlit as st

st.set_page_config(
    page_title="Performance Analysis Tool", page_icon="📊", layout="wide"
)


config_page = st.Page(
    "frontend/pages/config.py", title="Config", icon=":material/login:"
)
data_pipeline_page = st.Page(
    "frontend/pages/data_pipeline.py",
    title="Data Pipeline",
    icon=":material/transform:",
)
model_pipeline_page = st.Page(
    "frontend/pages/model_pipeline.py",
    title="Model Pipeline",
    icon=":material/transform:",
)
result_page = st.Page(
    "frontend/pages/results.py", title="Results", icon=":material/insights:"
)

pg = st.navigation(
    {
        "Config": [config_page],
        "Pipeline": [data_pipeline_page, model_pipeline_page],
        "Results": [result_page],
    }
)

pg.run()
