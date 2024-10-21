from abc import ABC, abstractmethod

import polars as pl


class MetricCalculator(ABC):
    """
    Abstract base class for metric calculators.
    """

    @abstractmethod
    def calculate(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Calculate metrics based on input data.

        Args:
            data (pl.LazyFrame): Input data for metric calculation.

        Returns:
            pl.LazyFrame: Data with calculated metrics added.
        """
        pass
