import streamlit as st

from src.config import Configuration
from src.data.preprocess.factory import PreprocessorStepFactory
from src.metrics.standardizer.factory import StandardizerFactory
from src.metrics.calculator.factory import MetricCalculatorFactory


def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "config" not in st.session_state:
        st.session_state.config = Configuration().model_dump()


# Initialize session state
initialize_session_state()

# Main content
st.title("Configuration")

# Data Source Section
st.header("Data Source")
data_source = st.selectbox(
    "Select Data Source",
    options=["parquet"],
    key="data_source",
)

data_file_path = st.text_input(
    "Enter Data file path",
    key="data_file_path",
    value=st.session_state.config["data_file_path"],
)


# Processing Section
st.header("Data Processing")
preprocessor_steps = st.multiselect(
    "Select preprocessor steps",
    options=list(PreprocessorStepFactory._steps.keys()),
    default=st.session_state.config["preprocessor_steps"],
    key="preprocessor_steps",
)

# Metrics Section
st.header("Metrics Configuration")
selected_metrics = st.multiselect(
    "Select metrics",
    options=MetricCalculatorFactory.get_registered_types(),
    default=st.session_state.config["selected_metrics"],
    key="metrics",
)

# Weights Section
if selected_metrics:
    with st.expander("Metric Weights"):
        metric_weights = {}
        for metric in selected_metrics:
            weight = st.slider(
                f"Weight for {metric}",
                0.0,
                1.0,
                value=st.session_state.config["metric_weights"].get(
                    metric, 1.0 / len(selected_metrics)
                ),
                key=f"weight_{metric}",
            )
            metric_weights[metric] = weight
        st.info(f"Total weight: {100*sum(metric_weights.values())}%")

# Standardization Section
st.header("Standardization")
standardizer = st.selectbox(
    "Select standardizer",
    options=StandardizerFactory.available_standardizers(),
    key="standardizer",
)

risk_free_rate = st.number_input(
    "Risk-free rate",
    value=st.session_state.config["risk_free_rate"],
    step=0.001,
    format="%.3f",
    key="risk_free_rate",
)

# Save Configuration
if st.button("Save Configuration"):
    if not data_file_path:
        st.error("Please enter a valid data file path.")
        raise ValueError("Invalid data file path")

    st.session_state.config = Configuration(
        data_source=data_source,
        data_file_path=data_file_path,
        preprocessor_steps=preprocessor_steps,
        selected_metrics=selected_metrics,
        metric_weights=metric_weights,
        standardizer=standardizer,
        risk_free_rate=risk_free_rate,
    ).model_dump()
    st.success("Configuration saved successfully!")

    with st.expander("Current Configuration"):
        st.json(st.session_state.config)
