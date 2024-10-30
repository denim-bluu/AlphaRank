import streamlit as st

from src.config import DataConfig, ModelConfig
from src.data.preprocess.factory import PreprocessorStepFactory
from src.standardizers.factory import StandardizerFactory
from src.metrics.factory import MetricCalculatorFactory
from src.weightings.factory import WeightingMethodFactory


def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "data_config" not in st.session_state:
        st.session_state.data_config = DataConfig().model_dump()
    if "model_config" not in st.session_state:
        st.session_state.model_config = ModelConfig().model_dump()


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
    value=st.session_state.data_config["data_file_path"],
)


# Processing Section
st.header("Data Processing")
preprocessor_steps = st.multiselect(
    "Select preprocessor steps",
    options=list(PreprocessorStepFactory._registry.keys()),
    default=st.session_state.data_config["preprocessor_steps"],
    key="preprocessor_steps",
)

# Metrics Section
st.header("Metrics Configuration")
selected_metrics = st.multiselect(
    "Select metrics",
    options=MetricCalculatorFactory.get_registered_types(),
    default=st.session_state.model_config["selected_metrics"],
    key="metrics",
)

# Weighting Method Section
st.header("Weighting Method")
weighting_method = st.selectbox(
    "Select weighting method",
    options=WeightingMethodFactory.get_registered_types(),
    key="weighting_method",
)

# Standardization Section
st.header("Standardization")
standardizer = st.selectbox(
    "Select standardizer",
    options=StandardizerFactory.get_registered_types(),
    key="standardizer",
)


# Save Configuration
if st.button("Save Configuration"):
    if not data_file_path:
        st.error("Please enter a valid data file path.")
        raise ValueError("Invalid data file path")

    st.session_state.data_config = DataConfig(
        data_source=data_source,
        data_file_path=data_file_path,
        preprocessor_steps=preprocessor_steps,
    ).model_dump()
    st.session_state.model_config = ModelConfig(
        selected_metrics=selected_metrics,
        weighting_method=weighting_method,
        standardizer=standardizer,
    ).model_dump()
    st.success("Configuration saved successfully!")

    with st.expander("Current Data Configuration"):
        st.json(st.session_state.data_config)
    with st.expander("Current Model Configuration"):
        st.json(st.session_state.model_config)
