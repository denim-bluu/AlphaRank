import plotly.express as px
import streamlit as st

from src.pipeline import ModelPipeline


st.header("Results")

if st.session_state.model_pipeline is None:
    st.stop()
model_pipeline: ModelPipeline = st.session_state.model_pipeline

scored_results = model_pipeline.pm_scores.sort("PMScore").collect()
metric_results = model_pipeline.metric_data.collect()

st.subheader("Portfolio Manager Rankings")
st.dataframe(scored_results)

st.subheader("Metric Results")
st.dataframe(metric_results)

st.subheader("Performance Metrics")
for metric in st.session_state.model_pipeline._metric_columns:
    fig = px.bar(
        metric_results, x="Strategy_ID", y=metric, title=f"{metric} by Strategy"
    )
    st.plotly_chart(fig)

st.subheader("Correlation Matrix")
corr_matrix = metric_results[st.session_state.model_pipeline._metric_columns].corr()
fig = px.imshow(corr_matrix, title="Metric Correlation Matrix")
st.plotly_chart(fig)
