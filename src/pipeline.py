from typing import Dict, Optional
from loguru import logger
from .config import ModelConfig
from .calculators.pipeline import CalculationPipeline
from .calculators.factory import MetricCalculatorFactory
from .aggregators.portfolio_aggregator.portfolio import PMScoreAggregator
from .standardizers.factory import StandardizerFactory
from .aggregators.strategy_aggregator.strategy import WeightedSumScoreAggregator
from .weightings.factory import WeightingMethodFactory
import polars as pl


class ModelPipeline:
    """
    ModelPipeline orchestrates the entire data processing and calculation workflow.

    Attributes:
        metric_data (pl.LazyFrame): Container for metric data.
        standardized_data (pl.LazyFrame): Container for standardized data.
        weights (Dict[str, float]): Dictionary of weights for metrics.
        strategy_scores (pl.LazyFrame): Container for strategy scores.
        pm_scores (pl.LazyFrame): Container for PM scores.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the ModelPipeline with the given configuration.

        Args:
            config (ModelConfig): Configuration object containing pipeline settings.
        """
        self._calculators = [
            MetricCalculatorFactory.create(m) for m in config.selected_metrics
        ]
        self._metric_columns = config.selected_metrics
        self._calculation_pipeline = CalculationPipeline(
            self._calculators, group_by_columns=["PM_ID", "Strategy_ID"]
        )
        self._metric_types = self._calculation_pipeline.get_calculator_types()
        self._standardizer = StandardizerFactory.create(config.standardizer)
        self._weighting_method = WeightingMethodFactory.create(config.weighting_method)
        self._score_aggregator = WeightedSumScoreAggregator()
        self._pm_score_aggregator = PMScoreAggregator()

        # Data containers
        self.metric_data: pl.LazyFrame
        self.standardized_data: pl.LazyFrame
        self.weights: Dict[str, float]
        self.strategy_scores: pl.LazyFrame
        self.pm_scores: pl.LazyFrame

    def run(
        self, data: pl.LazyFrame, manual_weights: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Run the entire pipeline and return the final PM score.

        Args:
            data (pl.LazyFrame): Input data to process.
            manual_weights (Optional[Dict[str, float]]): Manually set weights for metrics.
        """
        self.metric_data = self._calculate_metrics(data)
        self.standardized_data = self._standardize_metrics(self.metric_data)
        self.weights = self._calculate_weights(self.standardized_data, manual_weights)
        self.strategy_scores = self._aggregate_strategy_scores(
            self.standardized_data, self.weights
        )
        self.pm_scores = self._aggregate_pm_scores(self.strategy_scores)

    def _calculate_metrics(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Calculate metrics from the input data.

        Args:
            data (pl.LazyFrame): Input data to process.

        Returns:
            pl.LazyFrame: Calculated metric data.
        """
        logger.info("Starting metric calculation pipeline...")
        metric_data = self._calculation_pipeline.run(data)
        logger.info("Metric calculation pipeline completed.")
        return metric_data

    def _standardize_metrics(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Standardize the calculated metrics.

        Args:
            data (pl.LazyFrame): Calculated metric data.

        Returns:
            pl.LazyFrame: Standardized metric data.
        """
        logger.info("Starting standardization...")
        standardized_data = self._standardizer.standardize(data, self._metric_types)
        logger.info("Standardization completed.")
        return standardized_data

    def _calculate_weights(
        self, data: pl.LazyFrame, manual_weights: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate weights for the standardized metrics, or use manually set weights if provided.

        Args:
            data (pl.LazyFrame): Standardized metric data.
            manual_weights (Optional[Dict[str, float]]): Manually set weights for metrics.

        Returns:
            Dict[str, float]: Dictionary of weights for metrics.
        """
        if manual_weights:
            logger.info("Using manually set weights.")
            return manual_weights

        logger.info("Calculating weights...")
        weights = self._weighting_method.calculate_weights(data, self._metric_columns)
        logger.info("Weights calculation completed.")
        return weights

    def _aggregate_strategy_scores(
        self, data: pl.LazyFrame, weights: Dict[str, float]
    ) -> pl.LazyFrame:
        """
        Aggregate strategy scores using the calculated weights.

        Args:
            data (pl.LazyFrame): Standardized metric data.
            weights (Dict[str, float]): Dictionary of weights for metrics.

        Returns:
            pl.LazyFrame: Aggregated strategy scores.
        """
        logger.info("Aggregating strategy scores...")
        strategy_scores = self._score_aggregator.aggregate(
            data=data,
            metric_columns=self._metric_columns,
            weights=weights,
        )
        logger.info("Strategy scores aggregation completed.")
        return strategy_scores

    def _aggregate_pm_scores(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Aggregate PM scores from the strategy scores.

        Args:
            data (pl.LazyFrame): Aggregated strategy scores.

        Returns:
            pl.LazyFrame: Aggregated PM scores.
        """
        logger.info("Aggregating PM scores...")
        pm_scores = self._pm_score_aggregator.aggregate(data=data)
        logger.info("PM scores aggregation completed.")
        return pm_scores
