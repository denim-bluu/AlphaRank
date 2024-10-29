from pydantic import BaseModel, field_validator, Field

from src.data.preprocess.factory import PreprocessorStepFactory

from src.calculators.factory import MetricCalculatorFactory

from loguru import logger
from pathlib import Path

from src.standardizers.factory import StandardizerFactory
from src.weightings.factory import WeightingMethodFactory


logger.add("performance_analysis.log", rotation="500 MB", level="INFO")


class DataConfig(BaseModel):
    data_source: str = Field(
        default="parquet",
        description="Type of data source to use for the analysis.",
    )
    data_file_path: str = Field(
        default="data/mock_performance_data.parquet",
        description="Path to the data file.",
    )
    groupby_columns: list[str] = Field(
        default=["PM_ID", "Strategy_ID"],
        description="Columns to group the data by.",
    )
    preprocessor_steps: list[str] = Field(
        default=PreprocessorStepFactory.get_registered_types(),
        description="List of preprocessor steps to apply to the data.",
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
        if set(v) - set(PreprocessorStepFactory._registry.keys()):
            raise ValueError("Invalid preprocessor step.")
        return v


class ModelConfig(BaseModel):
    selected_metrics: list[str] = Field(
        default=MetricCalculatorFactory.get_registered_types(),
        description="List of metrics to calculate.",
    )
    standardizer: str = Field(
        default="MinMax",
        description="Type of standardizer to use for the metrics.",
    )
    weighting_method: str = Field(
        default="EntropyWeighting",
        description="Weights for each metric in the scoring.",
    )

    @field_validator("selected_metrics")
    def check_selected_metrics(cls, v):
        if not v:
            raise ValueError("Selected metrics must not be empty.")
        if set(v) - set(MetricCalculatorFactory.get_registered_types()):
            raise ValueError("Invalid metric.")
        return v

    @field_validator("standardizer")
    def check_standardizer(cls, v):
        if v not in StandardizerFactory.get_registered_types():
            raise ValueError("Invalid standardizer.")
        return v

    @field_validator("weighting_method")
    def check_weighting_method(cls, v):
        if v not in WeightingMethodFactory.get_registered_types():
            raise ValueError("Invalid weighting method.")
        return v
