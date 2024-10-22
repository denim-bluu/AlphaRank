from typing import Dict, List

import polars as pl
from loguru import logger

from .portfolio_aggregator.base import PortfolioScoreAggregator
from .standardizer.base import MetricStandardizer
from .strategy_aggregator.base import StrategyScoreAggregator


class ScoringPipeline:
    """Pipeline for standardizing metrics and aggregating them into a final score."""

    def __init__(
        self,
        standardizer: MetricStandardizer,
        strategy_aggregator: StrategyScoreAggregator,
        portfolio_aggregator: PortfolioScoreAggregator,
    ):
        self.standardizer = standardizer
        self.strategy_aggregator = strategy_aggregator
        self.portfolio_aggregator = portfolio_aggregator

    def run(
        self, data: pl.LazyFrame, metric_columns: List[str], weights: Dict[str, float]
    ) -> pl.LazyFrame:
        """
        Run the scoring pipeline.

        Args:
            data (pl.LazyFrame): Input data containing the single-value metrics for each strategy.
            metric_columns (List[str]): Names of the metric columns to process.
            weights (Dict[str, float]): Weights for each metric in the aggregation.

        Returns:
            pl.LazyFrame: Data with standardized metrics and aggregated score added.
        """
        try:
            logger.info("Standardizing metrics")
            standardized_data = self.standardizer.standardize(data, metric_columns)

            logger.info("Aggregating standardized metrics")
            standardized_columns = [f"{col}_standardized" for col in metric_columns]
            _weights = {
                f"{col}_standardized": weight for col, weight in weights.items()
            }
            scored_data = self.strategy_aggregator.aggregate(
                standardized_data, standardized_columns, _weights
            )

            scored_data = self.portfolio_aggregator.aggregate(scored_data)

            return scored_data

        except Exception as e:
            logger.exception(f"Error in scoring pipeline: {str(e)}")
            raise
