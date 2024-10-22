from abc import ABC, abstractmethod

import polars as pl


class PortfolioScoreAggregator(ABC):
    """Abstract base class for score aggregation"""

    @abstractmethod
    def aggregate(self, data: pl.LazyFrame) -> pl.LazyFrame:
        pass
