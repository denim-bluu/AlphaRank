import numpy as np
import polars as pl

from src.calculators.base import MetricCalculator, MetricType


class SharpeRatio(MetricCalculator):
    """Calculator for Sharpe ratio metric.

    Calculates the Sharpe ratio, which measures risk-adjusted return relative
    to the risk-free rate.

    Formula:
        Sharpe Ratio = (Mean(Return) - Risk_Free_Rate) / Std(Return) * sqrt(12)

    Attributes:
        required_columns (Set[str]): Required columns for calculation.
        risk_free_rate (float): Annual risk-free rate.

    Example:
        ```python
        calculator = SharpeRatioCalculator(risk_free_rate=0.02)
        result = calculator.calculate(data)
        ```
    """

    type: MetricType = MetricType.POSITIVE

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        """Initialize the SharpeRatioCalculator.

        Args:
            risk_free_rate: Annual risk-free rate (default: 2%).
            name: Optional custom name for the calculator.
        """
        self.risk_free_rate = risk_free_rate / 12  # Convert to monthly

    def expression(self) -> pl.Expr:
        """Calculate Sharpe ratio for the given data.

        Args:
            data: LazyFrame containing Return column.

        Returns:
            LazyFrame with added Sharpe_Ratio column.
        """
        return (
            (pl.col("Return").mean() - self.risk_free_rate)
            / pl.col("Return").std()
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
        required_columns (Set[str]): Required columns for calculation.

    Example:
        ```python
        calculator = InformationRatioCalculator()
        result = calculator.calculate(data)
        ```
    """

    type: MetricType = MetricType.POSITIVE

    def expression(self) -> pl.Expr:
        """Calculate Information ratio for the given data.

        Args:
            data: LazyFrame containing Return and Benchmark_Return columns.

        Returns:
            LazyFrame with added Information_Ratio column.
        """
        return (
            (pl.col("Return").mean() - pl.col("Benchmark_Return").mean())
            / (pl.col("Return") - pl.col("Benchmark_Return")).std()
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
        required_columns (Set[str]): Required columns for calculation.
        risk_free_rate (float): Annual risk-free rate.

    Example:
        ```python
        calculator = SortinoRatioCalculator(risk_free_rate=0.02)
        result = calculator.calculate(data)
        ```
    """

    type: MetricType = MetricType.POSITIVE

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        """Initialize the SortinoRatioCalculator.

        Args:
            risk_free_rate: Annual risk-free rate (default: 2%).
            name: Optional custom name for the calculator.
        """
        self.risk_free_rate = risk_free_rate / 12  # Convert to monthly

    def expression(self) -> pl.Expr:
        """Calculate Sortino ratio for the given data.

        Args:
            data: LazyFrame containing Return column.

        Returns:
            LazyFrame with added Sortino_Ratio column.
        """
        negative_returns = pl.when(pl.col("Return") < 0).then(pl.col("Return"))
        return (
            (pl.col("Return").mean() - self.risk_free_rate)
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
        required_columns (Set[str]): Required columns for calculation.
        threshold (float): Return threshold for separating gains and losses.

    Example:
        ```python
        calculator = OmegaRatioCalculator(threshold=0.0)
        result = calculator.calculate(data)
        ```
    """

    type: MetricType = MetricType.POSITIVE

    def __init__(self, threshold: float = 0.0) -> None:
        """Initialize the OmegaRatioCalculator.

        Args:
            threshold: Return threshold for separating gains and losses.
            name: Optional custom name for the calculator.
        """
        self.threshold = threshold

    def expression(self) -> pl.Expr:
        """Calculate Omega ratio for the given data.

        Args:
            data: LazyFrame containing Return and Benchmark_Return columns.

        Returns:
            LazyFrame with added Omega_Ratio column.
        """
        excess_returns_expr = pl.col("Return") - pl.col("Benchmark_Return")
        gains_expr = (
            pl.when(excess_returns_expr > self.threshold)
            .then(excess_returns_expr - self.threshold)
            .otherwise(0)
        )
        losses_expr = (
            pl.when(excess_returns_expr <= self.threshold)
            .then(self.threshold - excess_returns_expr)
            .otherwise(0)
        )
        return gains_expr.sum() / losses_expr.sum()
