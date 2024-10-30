from abc import ABC, abstractmethod

from enum import StrEnum
import pandas as pd


class MetricType(StrEnum):
    """Enum for different types of financial metrics."""

    POSITIVE = "Positive"
    NEGATIVE = "Negative"


class MetricCalculator(ABC):
    type: MetricType

    @abstractmethod
    def calculate_for_group(self, group_data: pd.DataFrame) -> float:
        """Calculate metric for a single group of data."""
        raise NotImplementedError

    def calculate_all_groups(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate metric for all groups in the dataset."""
        results = data.groupby(["PM_ID", "Strategy_ID"]).apply(self.calculate_for_group)
        return pd.DataFrame({self.__class__.__name__: results}).reset_index()
