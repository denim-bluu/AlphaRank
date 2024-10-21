from typing import Dict, List

import polars as pl
from loguru import logger

from .aggregator.base import MetricAggregator
from .standardizer.base import MetricStandardizer


class ScoringPipeline:
    """Pipeline for standardizing metrics and aggregating them into a final score."""

    def __init__(self, standardizer: MetricStandardizer, aggregator: MetricAggregator):
        self.standardizer = standardizer
        self.aggregator = aggregator

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

            # Check if the weights sum up to 1
            if sum(_weights.values()) != 1:
                logger.error("Weights must sum up to 1")
                raise ValueError("Weights must sum up to 1")

            # Check if the weights are positive
            if any(weight < 0 for weight in _weights.values()):
                logger.error("Weights must be positive")
                raise ValueError("Weights must be positive")

            # Check if columns are present in the data
            for col in standardized_columns:
                if col not in standardized_data.columns:
                    logger.error(f"Column {col} not found in the data")
                    raise ValueError(f"Column {col} not found in the data")

            scored_data = self.aggregator.aggregate(
                standardized_data, standardized_columns, _weights
            )

            return scored_data

        except Exception as e:
            logger.exception(f"Error in scoring pipeline: {str(e)}")
            raise
