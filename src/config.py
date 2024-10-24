from pydantic import BaseModel, field_validator, Field

from src.data.source import ParquetDataSource
from src.data.validator import DataValidator
from src.metrics.portfolio_aggregator.portfolio import PMScoreAggregator
from src.metrics.standardizer.factory import StandardizerFactory
from src.metrics.strategy_aggregator.strategy import WeightedSumScoreAggregator
from src.orchestrator import PerformanceAnalysisOrchestrator

from src.data.preprocessor import DataPreprocessor, PreprocessorStepFactory
from src.metrics.calculator.factory import MetricCalculatorFactory

from loguru import logger
from pathlib import Path

from src.data.pipeline import DataPipeline
from src.metrics.calculator.pipeline import CalculationPipeline

logger.add("performance_analysis.log", rotation="500 MB", level="INFO")


class Configuration(BaseModel):
    data_source: str = Field(
        default="parquet",
        description="Type of data source to use for the analysis.",
    )
    data_file_path: str = Field(
        default="data/mock_performance_data.parquet",
        description="Path to the data file.",
    )
    preprocessor_steps: list[str] = Field(
        default=PreprocessorStepFactory.available_steps(),
        description="List of preprocessor steps to apply to the data.",
    )
    selected_metrics: list[str] = Field(
        default=MetricCalculatorFactory.available_calculators(),
        description="List of metrics to calculate.",
    )
    standardizer: str = Field(
        default="zscore",
        description="Type of standardizer to use for the metrics.",
    )
    metric_weights: dict[str, float] = Field(
        default_factory=lambda: {
            metric: 1.0 / len(MetricCalculatorFactory.available_calculators())
            for metric in MetricCalculatorFactory.available_calculators()
        },
        description="Weights for each metric in the scoring.",
    )
    risk_free_rate: float = Field(
        default=0.02,
        description="Annual risk-free rate used in the Sharpe ratio calculation.",
    )

    @field_validator("data_source")
    def check_data_source(cls, v):
        if v not in ["parquet", "csv"]:
            raise ValueError("Invalid data source.")
        return v

    @field_validator("data_file_path")
    def check_data_file_path(cls, v):
        if not Path(v).is_file():
            raise ValueError("Data file does not exist.")
        return v

    @field_validator("preprocessor_steps")
    def check_preprocessor_steps(cls, v):
        if not v:
            raise ValueError("Preprocessor steps must not be empty.")
        if set(v) - set(PreprocessorStepFactory._steps.keys()):
            raise ValueError("Invalid preprocessor step.")
        return v

    @field_validator("selected_metrics")
    def check_selected_metrics(cls, v):
        if not v:
            raise ValueError("Selected metrics must not be empty.")
        if set(v) - set(MetricCalculatorFactory.available_calculators()):
            raise ValueError("Invalid metric.")
        return v

    @field_validator("standardizer")
    def check_standardizer(cls, v):
        if v not in StandardizerFactory._standardizers.keys():
            raise ValueError("Invalid standardizer.")
        return v

    @field_validator("metric_weights")
    def check_metric_weights(cls, v):
        if not v:
            raise ValueError("Metric weights must not be empty.")
        if not all(0 <= weight <= 1 for weight in v.values()):
            raise ValueError("Weights must be between 0 and 1.")
        if sum(v.values()) != 1:
            raise ValueError("Weights must sum to 1.")
        return v

    @field_validator("risk_free_rate")
    def check_risk_free_rate(cls, v):
        if v < 0:
            raise ValueError("Risk-free rate must be non-negative.")
        return v


def instantiate_orchestrator(config: Configuration) -> PerformanceAnalysisOrchestrator:
    data_source = ParquetDataSource(config.data_file_path)

    validator = DataValidator()
    preprocessor_steps = [
        PreprocessorStepFactory.create_step(step) for step in config.preprocessor_steps
    ]
    preprocessor = DataPreprocessor(preprocessor_steps)

    data_pipeline = DataPipeline(data_source, validator, preprocessor)

    factory = MetricCalculatorFactory()
    calculators = [factory.create_calculator(m) for m in config.selected_metrics]
    metric_pipeline = CalculationPipeline(calculators)

    standardizer = StandardizerFactory.create_standardizer(config.standardizer)
    strategy_aggregator = WeightedSumScoreAggregator()
    pm_score_aggregator = PMScoreAggregator()

    return PerformanceAnalysisOrchestrator(
        data_pipeline=data_pipeline,
        calculation_pipeline=metric_pipeline,
        standardizer=standardizer,
        strategy_aggregator=strategy_aggregator,
        portfolio_aggregator=pm_score_aggregator,
    )
