# File: pages/2_📈_analysis.py

import streamlit as st
from loguru import logger

from src.config import Configuration, instantiate_orchestrator
from src.orchestrator import PerformanceAnalysisOrchestrator


@st.cache_resource
def initialize_orchestrator(config: dict) -> PerformanceAnalysisOrchestrator:
    """Initialize the analysis orchestrator with configuration."""
    try:
        return instantiate_orchestrator(Configuration(**config))
    except Exception as e:
        logger.exception("Error initializing orchestrator")
        st.error(f"Error initializing analysis components: {str(e)}")
        raise e


# Page content starts here
st.title("Performance Analysis")
# Initialize orchestrator
orchestrator = initialize_orchestrator(st.session_state.config)
if orchestrator is None:
    st.stop()

# Analysis controls
st.header("Analysis Controls")

# View current configuration
with st.expander("Current Configuration"):
    st.json(st.session_state.config)


# Run analysis button
if st.button("Run Analysis", type="primary"):
    with st.spinner("Running analysis..."):
        try:
            # Run analysis
            orchestrator.run_analysis(
                metric_columns=st.session_state.config["selected_metrics"],
                weights=st.session_state.config["metric_weights"],
            )

            st.session_state.scored_results = orchestrator.get_scored_results()
            st.dataframe(st.session_state.scored_results)

        except Exception as e:
            logger.exception("Error during analysis")
            st.error(f"An error occurred during analysis: {str(e)}")
