from abc import ABC, abstractmethod
from typing import Dict, List

import polars as pl


class MetricAggregator(ABC):
    """Abstract base class for metric aggregators."""

    @abstractmethod
    def aggregate(
        self, data: pl.LazyFrame, metric_columns: List[str], weights: Dict[str, float]
    ) -> pl.LazyFrame:
        """
        Aggregate multiple metrics into a single score.

        Args:
            data (pl.LazyFrame): Input data containing the metrics to aggregate.
            metric_columns (List[str]): Names of the columns to aggregate.
            weights (Dict[str, float]): Weights for each metric in the aggregation.

        Returns:
            pl.LazyFrame: Data with the aggregated score added.
        """
        pass
