import streamlit as st
from frontend.utils import run_analysis


def render():
    st.header("Analysis")

    if st.button("Run Analysis", key="run_analysis_button"):
        try:
            results = run_analysis()
            st.session_state.results = results
            st.success("Analysis completed successfully!")
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
