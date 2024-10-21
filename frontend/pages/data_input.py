import streamlit as st

from frontend.utils import load_data


def render():
    st.header("Data Input")

    st.session_state.data_source = st.selectbox(
        "Select Data Source", ["parquet", "csv", "database"], key="data_source_select"
    )

    if st.session_state.data_source in ["parquet", "csv"]:
        st.session_state.file_path = st.text_input(
            "Enter file path", value=st.session_state.file_path, key="file_path_input"
        )

    if st.button("Load Data", key="load_data_button"):
        try:
            if not st.session_state.file_path:
                raise ValueError("File path is required!")
            data = load_data(st.session_state.data_source, st.session_state.file_path)
            st.success("Data loaded successfully!")
            st.write(data.head())
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
