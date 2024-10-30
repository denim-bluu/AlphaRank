from abc import ABC, abstractmethod

import pandas as pd


class PortfolioScoreAggregator(ABC):
    """Abstract base class for score aggregation"""

    @abstractmethod
    def aggregate(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
