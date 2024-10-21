import streamlit as st
from frontend.pages import data_input, configuration, analysis, results


def initialize_session_state():
    if "data_source" not in st.session_state:
        st.session_state.data_source = "parquet"
    if "file_path" not in st.session_state:
        st.session_state.file_path = "data/mock_performance_data.parquet"
    if "preprocessor_steps" not in st.session_state:
        st.session_state.preprocessor_steps = []
    if "selected_metrics" not in st.session_state:
        st.session_state.selected_metrics = []
    if "metric_weights" not in st.session_state:
        st.session_state.metric_weights = {}
    if "risk_free_rate" not in st.session_state:
        st.session_state.risk_free_rate = 0.03
    if "standardizer" not in st.session_state:
        st.session_state.standardizer = "zscore"
    if "results" not in st.session_state:
        st.session_state.results = None


def main():
    st.set_page_config(page_title="Performance Analysis Tool", layout="wide")
    st.title("Performance Analysis Tool")

    initialize_session_state()

    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Navigate", ["Data Input", "Configuration", "Analysis", "Results"]
    )

    if page == "Data Input":
        data_input.render()
    elif page == "Configuration":
        configuration.render()
    elif page == "Analysis":
        analysis.render()
    elif page == "Results":
        results.render()


if __name__ == "__main__":
    main()
