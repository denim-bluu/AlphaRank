from typing import Dict, List
from src.weightings.base import WeightingMethod

import polars as pl


class EqualWeighting(WeightingMethod):
    def calculate_weights(
        self, metric_data: pl.LazyFrame, metric_columns: List[str]
    ) -> Dict[str, float]:
        """
        Calculate equal weights for given metrics

        Args:
            metrics: List of metrics to consider

        Returns:
            Dictionary of metric weights
        """
        if len(metric_columns) == 0:
            raise ValueError("No metrics to calculate weights for")
        return {metric: 1 / len(metric_columns) for metric in metric_columns}
