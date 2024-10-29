import streamlit as st
from loguru import logger

from src.config import DataConfig
from src.data.pipeline import DataPipeline
from src.data.preprocess.factory import PreprocessorStepFactory
from src.data.preprocess.preprocessor import DataPreprocessor
from src.data.source import ParquetDataSource
from src.data.validator import DataValidator


@st.cache_resource
def initialize_data_pipeline(config: dict) -> DataPipeline:
    """Initialize the analysis orchestrator with configuration."""
    try:
        data_config = DataConfig(**config)
        data_source = ParquetDataSource(data_config.data_file_path)
        validator = DataValidator()
        preprocessor = DataPreprocessor(
            [
                PreprocessorStepFactory.create(step)
                for step in data_config.preprocessor_steps
            ]
        )
        return DataPipeline(data_source, validator, preprocessor)
    except Exception as e:
        logger.exception("Error initializing data pipeline")
        st.error(f"Error initializing analysis components: {str(e)}")
        raise e


st.title("Data Pipeline")
# Initialize Data Pipeline
data_pipeline = initialize_data_pipeline(st.session_state.data_config)

if data_pipeline is None:
    st.stop()
# View current configuration
with st.expander("Current Data Configuration", expanded=True):
    st.json(st.session_state.data_config)

# Run Data Pipeline
st.subheader("Data Pipeline")
if st.button("Run Data Pipeline", type="primary"):
    with st.spinner("Running data pipeline..."):
        try:
            st.session_state.data = data_pipeline.run()
            st.success("Data pipeline completed successfully.")
        except Exception as e:
            logger.exception("Error during data pipeline")
            st.error(f"An error occurred during data pipeline: {str(e)}")

# View data
with st.expander("View Data", expanded=False):
    st.write(st.session_state.data)
