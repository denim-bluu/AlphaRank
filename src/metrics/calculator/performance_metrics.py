from typing import Optional
import polars as pl

from .base import MetricCalculator


class ExcessReturnCalculator(MetricCalculator):
    """Calculator for excess return metric.

    Calculates the excess return, which is the difference between the return
    of an investment and the return of a benchmark index.

    Formula:
        Excess Return = Average(Return - Benchmark_Return)

    Attributes:
        name (str): Name of the calculator.
        required_columns (Set[str]): Required columns for calculation.

    Example:
        ```python
        calculator = ExcessReturnCalculator()
        result = calculator.calculate(data)
        ```
    """

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(required_columns={"Return", "Benchmark_Return"}, name=name)

    def _calculate_metric(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data.group_by(self.group_by_columns).agg(
            [
                ((pl.col("Return") - pl.col("Benchmark_Return")).mean()).alias(
                    "Excess_Return"
                )
            ]
        )


class BetaCalculator(MetricCalculator):
    """Calculator for beta metric.

    Calculates beta, which measures the volatility of a security or portfolio
    relative to the market. Used in CAPM to describe the relationship between
    expected return and systematic risk.

    Formula:
        Beta = Covariance(Return, Benchmark_Return) / Variance(Benchmark_Return)

    Attributes:
        name (str): Name of the calculator.
        required_columns (Set[str]): Required columns for calculation.

    Example:
        ```python
        calculator = BetaCalculator()
        result = calculator.calculate(data)
        ```
    """

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(required_columns={"Return", "Benchmark_Return"}, name=name)

    def _calculate_metric(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data.group_by(self.group_by_columns).agg(
            [
                (
                    pl.cov(pl.col("Return"), pl.col("Benchmark_Return"))
                    / pl.col("Benchmark_Return").var()
                ).alias("Beta")
            ]
        )
