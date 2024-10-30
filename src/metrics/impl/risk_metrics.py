import numpy as np
import pandas as pd
from src.metrics.base import MetricCalculator, MetricType


class Volatility(MetricCalculator):
    """Calculator for return volatility metric.

    Calculates the volatility (standard deviation) of returns for different strategies.
    Volatility is a statistical measure of the dispersion of returns for a given
    security or market index.

    Attributes:
        type (MetricType): Type of metric (NEGATIVE).
        annualize (bool): Whether to annualize the volatility (multiply by sqrt(12)).

    Example:
        ```python
        calculator = Volatility(annualize=True)
        result = calculator.calculate_for_group(group_data)
        ```
    """

    type = MetricType.NEGATIVE

    def __init__(self, annualize: bool = True) -> None:
        """Initialize the Volatility calculator.

        Args:
            annualize: Whether to annualize the volatility.
        """
        self.annualize = annualize

    def calculate_for_group(self, group_data: pd.DataFrame) -> float:
        """Calculate volatility for the given group of data.

        Args:
            group_data: DataFrame containing Return column.

        Returns:
            float: Calculated volatility.
        """
        volatility = group_data["Return"].std()
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
        type (MetricType): Type of metric (NEGATIVE).
        annualize (bool): Whether to annualize the tracking error.

    Example:
        ```python
        calculator = TrackingError()
        result = calculator.calculate_for_group(group_data)
        ```
    """

    type = MetricType.NEGATIVE

    def __init__(self, annualize: bool = True) -> None:
        """Initialize the TrackingError calculator.

        Args:
            annualize: Whether to annualize the tracking error.
        """
        self.annualize = annualize

    def calculate_for_group(self, group_data: pd.DataFrame) -> float:
        """Calculate tracking error for the given group of data.

        Args:
            group_data: DataFrame containing Return and Benchmark_Return columns.

        Returns:
            float: Calculated tracking error.
        """
        tracking_error = (group_data["Return"] - group_data["Benchmark_Return"]).std()

        if self.annualize:
            tracking_error = tracking_error * np.sqrt(12)
        return tracking_error


class ValueAtRisk(MetricCalculator):
    """Calculator for Value at Risk (VaR) metric.

    Calculates the Value at Risk (VaR), which measures the maximum potential loss
    over a given time horizon for a given confidence level.

    Formula:
        VaR = -quantile(Return, alpha)

    Attributes:
        type (MetricType): Type of metric (NEGATIVE).
        alpha (float): Confidence level (e.g., 0.05 for 95% confidence).

    Example:
        ```python
        calculator = ValueAtRisk(alpha=0.05)
        result = calculator.calculate_for_group(group_data)
        ```
    """

    type = MetricType.NEGATIVE

    def __init__(self, alpha: float = 0.05) -> None:
        """Initialize the ValueAtRisk calculator.

        Args:
            alpha: Confidence level (default: 0.05).
        """
        self.alpha = alpha

    def calculate_for_group(self, group_data: pd.DataFrame) -> float:
        """Calculate Value at Risk for the given group of data.

        Args:
            group_data: DataFrame containing Return column.

        Returns:
            float: Calculated Value at Risk.
        """
        return -group_data["Return"].quantile(self.alpha)
