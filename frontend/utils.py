import polars as pl
import streamlit as st

from src.data.preprocess import steps as pp
from src.data.pipeline import DataPipeline
import src.data.preprocess.preprocessor
from src.data.source import ParquetDataSource
from src.data.validator import DataValidator
from src.metrics.calculator.factory import MetricCalculatorFactory
from src.metrics.calculator.pipeline import CalculationPipeline
from src.metrics.portfolio_aggregator.portfolio import PMScoreAggregator
from src.metrics.standardizer.factory import StandardizerFactory
from src.metrics.strategy_aggregator.strategy import WeightedSumScoreAggregator
from src.orchestrator import PerformanceAnalysisOrchestrator


def load_data(source_type: str, file_path: str) -> pl.DataFrame:
    if source_type == "parquet":
        return pl.read_parquet(file_path)
    elif source_type == "csv":
        raise NotImplementedError("CSV data source not implemented yet.")
    else:
        raise ValueError(f"Unsupported data source type: {source_type}")


@st.cache_data
def run_analysis() -> PerformanceAnalysisOrchestrator:
    # Set up data source
    if st.session_state.data_source == "parquet":
        data_source = ParquetDataSource(st.session_state.file_path)
    elif st.session_state.data_source == "csv":
        raise NotImplementedError("CSV data source not implemented yet.")
    else:
        raise ValueError(f"Unsupported data source: {st.session_state.data_source}")

    # Set up validator and preprocessor
    validator = DataValidator()
    preprocessor_steps = [
        getattr(pp, step)() for step in st.session_state.preprocessor_steps
    ]
    preprocessor = src.data.preprocess.preprocessor.DataPreprocessor(preprocessor_steps)

    # Set up data pipeline
    data_pipeline = DataPipeline(data_source, validator, preprocessor)

    # Set up metric pipeline
    factory = MetricCalculatorFactory()
    calculators = [
        factory.create(metric.lower()) for metric in st.session_state.selected_metrics
    ]
    if "sharpe_ratio" in st.session_state.selected_metrics:
        calculators.append(
            factory.create(
                "sharpe_ratio", risk_free_rate=st.session_state.risk_free_rate
            )
        )
    metric_pipeline = CalculationPipeline(calculators)

    # Set up scoring pipeline
    standardizer = StandardizerFactory.create_standardizer(
        st.session_state.standardizer
    )
    strategy_aggregator = WeightedSumScoreAggregator()
    pm_score_aggregator = PMScoreAggregator()

    # Set up and run orchestrator
    orchestrator = PerformanceAnalysisOrchestrator(
        data_pipeline=data_pipeline,
        calculation_pipeline=metric_pipeline,
        standardizer=standardizer,
        strategy_aggregator=strategy_aggregator,
        portfolio_aggregator=pm_score_aggregator,
    )

    orchestrator.run_analysis(
        st.session_state.selected_metrics, st.session_state.metric_weights
    )
    return orchestrator
