from typing import Optional
import numpy as np
import polars as pl

from .base import MetricCalculator


class SharpeRatioCalculator(MetricCalculator):
    """Calculator for Sharpe ratio metric.

    Calculates the Sharpe ratio, which measures risk-adjusted return relative
    to the risk-free rate.

    Formula:
        Sharpe Ratio = (Mean(Return) - Risk_Free_Rate) / Std(Return) * sqrt(12)

    Attributes:
        name (str): Name of the calculator.
        required_columns (Set[str]): Required columns for calculation.
        risk_free_rate (float): Annual risk-free rate.

    Example:
        ```python
        calculator = SharpeRatioCalculator(risk_free_rate=0.02)
        result = calculator.calculate(data)
        ```
    """

    def __init__(
        self, risk_free_rate: float = 0.02, name: Optional[str] = None
    ) -> None:
        """Initialize the SharpeRatioCalculator.

        Args:
            risk_free_rate: Annual risk-free rate (default: 2%).
            name: Optional custom name for the calculator.
        """
        super().__init__(required_columns={"Return"}, name=name)
        self.risk_free_rate = risk_free_rate / 12  # Convert to monthly

    def _calculate_metric(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """Calculate Sharpe ratio for the given data.

        Args:
            data: LazyFrame containing Return column.

        Returns:
            LazyFrame with added Sharpe_Ratio column.
        """
        return data.group_by(self.group_by_columns).agg(
            [
                (
                    (pl.col("Return").mean() - self.risk_free_rate)
                    / pl.col("Return").std()
                    * np.sqrt(12)
                ).alias("Sharpe_Ratio")
            ]
        )


class InformationRatioCalculator(MetricCalculator):
    """Calculator for Information ratio metric.

    Calculates the Information ratio, which measures risk-adjusted excess return
    relative to a benchmark.

    Formula:
        IR = (Mean(Return) - Mean(Benchmark_Return)) /
             Std(Return - Benchmark_Return) * sqrt(12)

    Attributes:
        name (str): Name of the calculator.
        required_columns (Set[str]): Required columns for calculation.

    Example:
        ```python
        calculator = InformationRatioCalculator()
        result = calculator.calculate(data)
        ```
    """

    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize the InformationRatioCalculator.

        Args:
            name: Optional custom name for the calculator.
        """
        super().__init__(required_columns={"Return", "Benchmark_Return"}, name=name)

    def _calculate_metric(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """Calculate Information ratio for the given data.

        Args:
            data: LazyFrame containing Return and Benchmark_Return columns.

        Returns:
            LazyFrame with added Information_Ratio column.
        """
        return data.group_by(self.group_by_columns).agg(
            [
                (
                    (pl.col("Return").mean() - pl.col("Benchmark_Return").mean())
                    / (pl.col("Return") - pl.col("Benchmark_Return")).std()
                    * np.sqrt(12)
                ).alias("Information_Ratio")
            ]
        )


class SortinoRatioCalculator(MetricCalculator):
    """Calculator for Sortino ratio metric.

    Calculates the Sortino ratio, which measures risk-adjusted return using
    only downside deviation.

    Formula:
        Sortino Ratio = (Mean(Return) - Risk_Free_Rate) /
                       Std(Negative_Returns) * sqrt(12)

    Attributes:
        name (str): Name of the calculator.
        required_columns (Set[str]): Required columns for calculation.
        risk_free_rate (float): Annual risk-free rate.

    Example:
        ```python
        calculator = SortinoRatioCalculator(risk_free_rate=0.02)
        result = calculator.calculate(data)
        ```
    """

    def __init__(
        self, risk_free_rate: float = 0.02, name: Optional[str] = None
    ) -> None:
        """Initialize the SortinoRatioCalculator.

        Args:
            risk_free_rate: Annual risk-free rate (default: 2%).
            name: Optional custom name for the calculator.
        """
        super().__init__(required_columns={"Return"}, name=name)
        self.risk_free_rate = risk_free_rate / 12  # Convert to monthly

    def _calculate_metric(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """Calculate Sortino ratio for the given data.

        Args:
            data: LazyFrame containing Return column.

        Returns:
            LazyFrame with added Sortino_Ratio column.
        """
        negative_returns = pl.when(pl.col("Return") < 0).then(pl.col("Return"))
        return data.group_by(self.group_by_columns).agg(
            [
                (
                    (pl.col("Return").mean() - self.risk_free_rate)
                    / negative_returns.std()
                    * np.sqrt(12)
                ).alias("Sortino_Ratio")
            ]
        )


class OmegaRatioCalculator(MetricCalculator):
    """Calculator for Omega ratio metric.

    Calculates the Omega ratio, which evaluates the probability-weighted
    ratio of gains versus losses relative to a threshold.

    Formula:
        Omega Ratio = Sum(Gains above threshold) / Sum(Losses below threshold)

    Attributes:
        name (str): Name of the calculator.
        required_columns (Set[str]): Required columns for calculation.
        threshold (float): Return threshold for separating gains and losses.

    Example:
        ```python
        calculator = OmegaRatioCalculator(threshold=0.0)
        result = calculator.calculate(data)
        ```
    """

    def __init__(self, threshold: float = 0.0, name: Optional[str] = None) -> None:
        """Initialize the OmegaRatioCalculator.

        Args:
            threshold: Return threshold for separating gains and losses.
            name: Optional custom name for the calculator.
        """
        super().__init__(required_columns={"Return", "Benchmark_Return"}, name=name)
        self.threshold = threshold

    def _calculate_metric(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """Calculate Omega ratio for the given data.

        Args:
            data: LazyFrame containing Return and Benchmark_Return columns.

        Returns:
            LazyFrame with added Omega_Ratio column.
        """
        excess_returns = pl.col("Return") - pl.col("Benchmark_Return")
        gains = (
            pl.when(excess_returns > self.threshold)
            .then(excess_returns - self.threshold)
            .otherwise(0)
        )
        losses = (
            pl.when(excess_returns <= self.threshold)
            .then(self.threshold - excess_returns)
            .otherwise(0)
        )

        return (
            data.with_columns(gains.alias("Gains"), losses.alias("Losses"))
            .group_by(self.group_by_columns)
            .agg(
                [(pl.col("Gains").sum() / pl.col("Losses").sum()).alias("Omega_Ratio")]
            )
        )
