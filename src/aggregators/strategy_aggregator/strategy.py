from typing import Dict

import pandas as pd

from .base import StrategyScoreAggregator


class WeightedSumScoreAggregator(StrategyScoreAggregator):
    """Aggregator that uses a weighted sum of metrics."""

    def _aggregate(
        self, metric_data: pd.DataFrame, weights: Dict[str, float]
    ) -> pd.DataFrame:
        weighted_sum = sum(
            metric_data[metric] * weight for metric, weight in weights.items()
        )
        metric_data["StrategyScore"] = weighted_sum
        return metric_data
