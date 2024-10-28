from typing import Dict, List

import polars as pl

from .base import StrategyScoreAggregator


class WeightedSumScoreAggregator(StrategyScoreAggregator):
    """Aggregator that uses a weighted sum of metrics."""

    def expr(self, metric_columns: List[str], weights: Dict[str, float]) -> pl.Expr:
        weighted_sum_expr = sum(
            pl.col(col).mul(weights.get(col, 0.0)) for col in metric_columns
        )
        if not isinstance(weighted_sum_expr, pl.Expr):
            weighted_sum_expr = pl.lit(weighted_sum_expr)
        return weighted_sum_expr
