import plotly.express as px
import streamlit as st


def render():
    st.header("Results")

    if st.session_state.results is not None:
        results = st.session_state.results

        st.subheader("Strategy Rankings")
        st.dataframe(results.sort("rank"))

        st.subheader("Performance Metrics")
        for metric in st.session_state.selected_metrics:
            fig = px.bar(
                results, x="Strategy_ID", y=metric, title=f"{metric} by Strategy"
            )
            st.plotly_chart(fig)

        st.subheader("Correlation Matrix")
        corr_matrix = results[st.session_state.selected_metrics].corr()
        fig = px.imshow(corr_matrix, title="Metric Correlation Matrix")
        st.plotly_chart(fig)
    else:
        st.info("No results to display. Please run the analysis first.")
