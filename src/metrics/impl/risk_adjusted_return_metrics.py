import numpy as np
import pandas as pd

from src.metrics.base import MetricCalculator, MetricType


class SharpeRatio(MetricCalculator):
    """Calculator for Sharpe ratio metric.

    Calculates the Sharpe ratio, which measures risk-adjusted return relative
    to the risk-free rate.

    Formula:
        Sharpe Ratio = (Mean(Return) - Risk_Free_Rate) / Std(Return) * sqrt(12)

    Attributes:
        type (MetricType): Type of metric (POSITIVE).
        risk_free_rate (float): Annual risk-free rate.

    Example:
        ```python
        calculator = SharpeRatio(risk_free_rate=0.02)
        result = calculator.calculate_for_group(group_data)
        ```
    """

    type = MetricType.POSITIVE

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        """Initialize the SharpeRatio calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default: 2%).
        """
        self.risk_free_rate = risk_free_rate / 12  # Convert to monthly

    def calculate_for_group(self, group_data: pd.DataFrame) -> float:
        """Calculate Sharpe ratio for the given group of data.

        Args:
            group_data: DataFrame containing Return column.

        Returns:
            float: Calculated Sharpe ratio.
        """
        return (
            (group_data["Return"].mean() - self.risk_free_rate)
            / group_data["Return"].std()
            * np.sqrt(12)
        )


class InformationRatio(MetricCalculator):
    """Calculator for Information ratio metric.

    Calculates the Information ratio, which measures risk-adjusted excess return
    relative to a benchmark.

    Formula:
        IR = (Mean(Return) - Mean(Benchmark_Return)) /
             Std(Return - Benchmark_Return) * sqrt(12)

    Attributes:
        type (MetricType): Type of metric (POSITIVE).

    Example:
        ```python
        calculator = InformationRatio()
        result = calculator.calculate_for_group(group_data)
        ```
    """

    type = MetricType.POSITIVE

    def calculate_for_group(self, group_data: pd.DataFrame) -> float:
        """Calculate Information ratio for the given group of data.

        Args:
            group_data: DataFrame containing Return and Benchmark_Return columns.

        Returns:
            float: Calculated Information ratio.
        """
        return (
            (group_data["Return"].mean() - group_data["Benchmark_Return"].mean())
            / (group_data["Return"] - group_data["Benchmark_Return"]).std()
            * np.sqrt(12)
        )


class SortinoRatio(MetricCalculator):
    """Calculator for Sortino ratio metric.

    Calculates the Sortino ratio, which measures risk-adjusted return using
    only downside deviation.

    Formula:
        Sortino Ratio = (Mean(Return) - Risk_Free_Rate) /
                       Std(Negative_Returns) * sqrt(12)

    Attributes:
        type (MetricType): Type of metric (POSITIVE).
        risk_free_rate (float): Annual risk-free rate.

    Example:
        ```python
        calculator = SortinoRatio(risk_free_rate=0.02)
        result = calculator.calculate_for_group(group_data)
        ```
    """

    type = MetricType.POSITIVE

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        """Initialize the SortinoRatio calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default: 2%).
        """
        self.risk_free_rate = risk_free_rate / 12

    def calculate_for_group(self, group_data: pd.DataFrame) -> float:
        """Calculate Sortino ratio for the given group of data.

        Args:
            group_data: DataFrame containing Return column.

        Returns:
            float: Calculated Sortino ratio.
        """
        negative_returns = group_data[group_data["Return"] < 0]["Return"]
        return (
            (group_data["Return"].mean() - self.risk_free_rate)
            / negative_returns.std()
            * np.sqrt(12)
        )


class OmegaRatio(MetricCalculator):
    """Calculator for Omega ratio metric.

    Calculates the Omega ratio, which evaluates the probability-weighted
    ratio of gains versus losses relative to a threshold.

    Formula:
        Omega Ratio = Sum(Gains above threshold) / Sum(Losses below threshold)

    Attributes:
        type (MetricType): Type of metric (POSITIVE).
        threshold (float): Return threshold for separating gains and losses.

    Example:
        ```python
        calculator = OmegaRatio(threshold=0.0)
        result = calculator.calculate_for_group(group_data)
        ```
    """

    type = MetricType.POSITIVE

    def __init__(self, threshold: float = 0.0) -> None:
        """Initialize the OmegaRatio calculator.

        Args:
            threshold: Return threshold for separating gains and losses.
        """
        self.threshold = threshold

    def calculate_for_group(self, group_data: pd.DataFrame) -> float:
        """Calculate Omega ratio for the given group of data.

        Args:
            group_data: DataFrame containing Return and Benchmark_Return columns.

        Returns:
            float: Calculated Omega ratio.
        """
        excess_returns = group_data["Return"] - group_data["Benchmark_Return"]
        gains = excess_returns[excess_returns > self.threshold] - self.threshold
        losses = self.threshold - excess_returns[excess_returns <= self.threshold]
        return gains.sum() / losses.sum()
