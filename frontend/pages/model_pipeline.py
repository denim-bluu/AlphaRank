import streamlit as st
from loguru import logger

from src.config import ModelConfig
from src.pipeline import ModelPipeline


@st.cache_resource
def initialize_model_pipeline(config: dict) -> ModelPipeline:
    """Initialize the analysis orchestrator with configuration."""
    try:
        model_config = ModelConfig(**config)
        return ModelPipeline(model_config)
    except Exception as e:
        logger.exception("Error initializing model pipeline")
        st.error(f"Error initializing analysis components: {str(e)}")
        raise e


# Page content starts here
st.title("Performance Analysis")

# Initialize model pipeline
st.session_state.model_pipeline = initialize_model_pipeline(
    st.session_state.model_config
)
if st.session_state.model_pipeline is None:
    st.stop()

# Analysis controls
st.header("Analysis Controls")

with st.expander("Current Model Configuration", expanded=True):
    st.json(st.session_state.model_config)

# Run Model Pipeline step by step
st.subheader("Model Pipeline")

# Allow manual weights (Dictionary of Metric: Weight)
manual_override = st.checkbox("Manual Weights Override")
if manual_override:
    with st.expander("Metric Weights"):
        metric_weights = {}
        for metric in st.session_state.model_pipeline._metric_columns:
            weight = st.slider(
                f"Weight for {metric}",
                0.0,
                1.0,
                value=1.0 / len(st.session_state.model_pipeline._metric_columns),
                key=f"weight_{metric}",
            )
            metric_weights[metric] = weight
        st.info(f"Total weight: {100*sum(metric_weights.values())}%")
else:
    metric_weights = None

# Run Model Pipeline
if st.button("Run Model Pipeline", type="primary"):
    with st.spinner("Running model pipeline..."):
        try:
            if manual_override:
                st.session_state.model_pipeline.run(
                    st.session_state.data, metric_weights
                )
            else:
                st.session_state.model_pipeline.run(st.session_state.data)
            st.success("Model pipeline completed successfully.")

            # Show Weights
            with st.expander("Metric Weights", expanded=True):
                st.json(st.session_state.model_pipeline.weights)

        except Exception as e:
            logger.exception("Error during model pipeline")
            st.error(f"An error occurred during model pipeline: {str(e)}")
