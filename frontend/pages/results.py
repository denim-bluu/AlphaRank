import plotly.express as px
import streamlit as st

from src.orchestrator import PerformanceAnalysisOrchestrator


def render():
    st.header("Results")

    if st.session_state.results is not None:
        results: PerformanceAnalysisOrchestrator = st.session_state.results

        scored_results = results.get_scored_results().sort("Rank")
        metric_results = results.get_metric_results()

        st.subheader("Portfolio Manager Rankings")
        st.dataframe(scored_results)

        st.subheader("Metric Results")
        st.dataframe(metric_results)

        st.subheader("Performance Metrics")
        for metric in st.session_state.selected_metrics:
            fig = px.bar(
                metric_results, x="Strategy_ID", y=metric, title=f"{metric} by Strategy"
            )
            st.plotly_chart(fig)

        st.subheader("Correlation Matrix")
        corr_matrix = metric_results[st.session_state.selected_metrics].corr()
        fig = px.imshow(corr_matrix, title="Metric Correlation Matrix")
        st.plotly_chart(fig)
    else:
        st.info("No results to display. Please run the analysis first.")
