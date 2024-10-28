from abc import ABC, abstractmethod

import polars as pl
from enum import StrEnum


class MetricType(StrEnum):
    """Enum for different types of financial metrics."""

    POSITIVE = "Positive"
    NEGATIVE = "Negative"


class MetricCalculator(ABC):
    type: MetricType

    @abstractmethod
    def expression(self) -> pl.Expr:
        """Define the Polars expression for the metric calculation."""
        raise NotImplementedError
