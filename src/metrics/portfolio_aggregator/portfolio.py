import polars as pl

from .base import PortfolioScoreAggregator


class PMScoreAggregator(PortfolioScoreAggregator):
    """Aggregator that uses a weighted sum of metrics."""

    def aggregate(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return (
            data.group_by("PM_ID")
            .agg([pl.col("Strategy_Score").mean().alias("PM_Score")])
            .sort("PM_Score", descending=False)
        )
