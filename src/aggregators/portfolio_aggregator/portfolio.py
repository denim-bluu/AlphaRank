from .base import PortfolioScoreAggregator

import pandas as pd


class PMScoreAggregator(PortfolioScoreAggregator):
    """Aggregator that uses a weighted sum of metrics."""

    def aggregate(self, data: pd.DataFrame) -> pd.DataFrame:
        return (
            data.groupby("PM_ID")
            .agg({"StrategyScore": "mean"})
            .sort_values("StrategyScore", ascending=True)
            .rename(columns={"StrategyScore": "PMScore"})
        )
