from abc import ABC, abstractmethod
from typing import Dict
import polars as pl

from src.calculators.base import MetricType


class MetricStandardizer(ABC):
    """Abstract base class for metric standardizers."""

    @abstractmethod
    def standardize(
        self, data: pl.LazyFrame, metric_types: Dict[str, MetricType]
    ) -> pl.LazyFrame:
        """Standardize the given data based on the metric types."""
        pass
