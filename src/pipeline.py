from typing import Dict
from loguru import logger
from .config import Configuration
from .data.pipeline import DataPipeline
from .data.preprocess.factory import PreprocessorStepFactory
from .data.preprocess.preprocessor import DataPreprocessor
from .data.source import ParquetDataSource
from .data.validator import DataValidator
from .calculators.pipeline import CalculationPipeline
from .calculators.factory import MetricCalculatorFactory
from .aggregators.portfolio_aggregator.portfolio import PMScoreAggregator
from .standardizers.factory import StandardizerFactory
from .aggregators.strategy_aggregator.strategy import WeightedSumScoreAggregator
from .weightings.factory import WeightingMethodFactory
import polars as pl


class MainPipeline:
    """
    MainPipeline orchestrates the entire data processing and calculation workflow.

    Attributes:
        data (pl.LazyFrame): Container for raw data.
        metric_data (pl.LazyFrame): Container for metric data.
        standardized_data (pl.LazyFrame): Container for standardized data.
        weights (Dict[str, float]): Dictionary of weights for metrics.
        strategy_scores (pl.LazyFrame): Container for strategy scores.
    """

    def __init__(self, config: Configuration):
        """
        Initialize the MainPipeline with the given configuration.

        Args:
            config (Configuration): Configuration object containing pipeline settings.
        """
        self._data_source = ParquetDataSource(config.data_file_path)
        self._validator = DataValidator()
        self._preprocessor = DataPreprocessor(
            [PreprocessorStepFactory.create(step) for step in config.preprocessor_steps]
        )
        self._data_pipeline = DataPipeline(
            self._data_source, self._validator, self._preprocessor
        )
        self._calculators = [
            MetricCalculatorFactory.create(m) for m in config.selected_metrics
        ]
        self._metric_columns = config.selected_metrics
        self._calculation_pipeline = CalculationPipeline(
            self._calculators,
            group_by_columns=config.groupby_columns,
        )
        self._metric_types = self._calculation_pipeline.get_calculator_types()
        self._standardizer = StandardizerFactory.create(config.standardizer)
        self._weighting_method = WeightingMethodFactory.create(config.weighting_method)
        self._score_aggregator = WeightedSumScoreAggregator()
        self._pm_score_aggregator = PMScoreAggregator()

        # Data containers
        self.data: pl.LazyFrame
        self.metric_data: pl.LazyFrame
        self.standardized_data: pl.LazyFrame
        self.weights: Dict[str, float]
        self.strategy_scores: pl.LazyFrame

    def run(self) -> pl.LazyFrame:
        """
        Run the entire pipeline and return the final PM score.

        Returns:
            pl.LazyFrame: Final PM score.
        """
        logger.info("Starting data pipeline...")
        self.data = self._data_pipeline.run()
        logger.info("Data pipeline completed.")

        logger.info("Starting metric calculation pipeline...")
        self.metric_data = self._calculation_pipeline.run(self.data)
        logger.info("Metric calculation pipeline completed.")

        logger.info("Starting standardization...")
        self.standardized_data = self._standardizer.standardize(
            self.metric_data, self._metric_types
        )
        logger.info("Standardization completed.")

        logger.info("Calculating weights...")
        self.weights = self._weighting_method.calculate_weights(
            self.standardized_data, self._metric_columns
        )
        logger.info("Weights calculation completed.")

        logger.info("Aggregating strategy scores...")
        self.strategy_scores = self._score_aggregator.aggregate(
            data=self.standardized_data,
            metric_columns=self._metric_columns,
            weights=self.weights,
        )
        logger.info("Strategy scores aggregation completed.")

        logger.info("Aggregating PM scores...")
        pm_score = self._pm_score_aggregator.aggregate(data=self.strategy_scores)
        logger.info("PM scores aggregation completed.")

        return pm_score
