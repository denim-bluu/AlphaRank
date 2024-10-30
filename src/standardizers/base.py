from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd
from src.metrics.base import MetricType


class MetricStandardizer(ABC):
    """Abstract base class for metric standardizers."""

    @abstractmethod
    def standardize(
        self, data: pd.DataFrame, metric_types: Dict[str, MetricType]
    ) -> pd.DataFrame:
        """Standardize the given data based on the metric types."""
        pass
