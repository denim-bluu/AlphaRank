from typing import Dict, List

import polars as pl

from .base import StrategyScoreAggregator


class WeightedSumScoreAggregator(StrategyScoreAggregator):
    """Aggregator that uses a weighted sum of metrics."""

    def _aggregate(
        self, data: pl.LazyFrame, metric_columns: List[str], weights: Dict[str, float]
    ) -> pl.LazyFrame:
        weighted_sum_expr = sum(
            pl.col(col).mul(weights.get(col, 0.0)) for col in metric_columns
        )
        if not isinstance(weighted_sum_expr, pl.Expr):
            weighted_sum_expr = pl.lit(weighted_sum_expr)
        return data.with_columns([weighted_sum_expr.alias("Strategy_Score")])
