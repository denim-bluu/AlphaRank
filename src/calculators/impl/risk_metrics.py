import numpy as np
import polars as pl

from src.calculators.base import MetricCalculator, MetricType


class Volatility(MetricCalculator):
    """Calculator for return volatility metric.

    Calculates the volatility (standard deviation) of returns for different strategies.
    Volatility is a statistical measure of the dispersion of returns for a given
    security or market index.

    Attributes:
        required_columns (Set[str]): Required columns for calculation.
        annualize (bool): Whether to annualize the volatility (multiply by sqrt(12)).

    Example:
        ```python
        calculator = VolatilityCalculator(annualize=True)
        result = calculator.calculate(data)
        ```
    """

    type: MetricType = MetricType.NEGATIVE

    def __init__(self, annualize: bool = True) -> None:
        """Initialize the VolatilityCalculator.

        Args:
            annualize: Whether to annualize the volatility.
            name: Optional custom name for the calculator.
        """
        self.annualize = annualize

    def expression(self) -> pl.Expr:
        """Calculate volatility for the given data.

        Args:
            data: LazyFrame containing Return column.

        Returns:
            LazyFrame with added Volatility column.
        """
        volatility = pl.col("Return").std()
        if self.annualize:
            volatility = volatility * np.sqrt(12)

        return volatility


class TrackingError(MetricCalculator):
    """Calculator for tracking error metric.

    Calculates the tracking error, which measures how closely a portfolio follows
    its benchmark. It is computed as the standard deviation of the difference
    between portfolio and benchmark returns.

    Formula:
        Tracking Error = sqrt(12) * std(Return - Benchmark_Return)

    Attributes:
        required_columns (Set[str]): Required columns for calculation.
        annualize (bool): Whether to annualize the tracking error.

    Example:
        ```python
        calculator = TrackingErrorCalculator()
        result = calculator.calculate(data)
        ```
    """

    type: MetricType = MetricType.NEGATIVE

    def __init__(self, annualize: bool = True) -> None:
        """Initialize the TrackingErrorCalculator.

        Args:
            annualize: Whether to annualize the tracking error.
            name: Optional custom name for the calculator.
        """
        self.annualize = annualize

    def expression(self) -> pl.Expr:
        """Calculate tracking error for the given data.

        Args:
            data: LazyFrame containing Return and Benchmark_Return columns.

        Returns:
            LazyFrame with added Tracking_Error column.
        """
        tracking_error = (pl.col("Return") - pl.col("Benchmark_Return")).std()
        if self.annualize:
            tracking_error = tracking_error * np.sqrt(12)

        return tracking_error
