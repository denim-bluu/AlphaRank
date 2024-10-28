import polars as pl

from src.calculators.base import MetricCalculator, MetricType


class ExcessReturn(MetricCalculator):
    """Calculator for excess return metric.

    Calculates the excess return, which is the difference between the return
    of an investment and the return of a benchmark index.

    Formula:
        Excess Return = Average(Return - Benchmark_Return)

    Attributes:
        required_columns (Set[str]): Required columns for calculation.

    Example:
        ```python
        calculator = ExcessReturnCalculator()
        result = calculator.calculate(data)
        ```
    """

    type: MetricType = MetricType.POSITIVE

    def expression(self) -> pl.Expr:
        return (pl.col("Return") - pl.col("Benchmark_Return")).mean()


class Beta(MetricCalculator):
    """Calculator for beta metric.

    Calculates beta, which measures the volatility of a security or portfolio
    relative to the market. Used in CAPM to describe the relationship between
    expected return and systematic risk.

    Formula:
        Beta = Covariance(Return, Benchmark_Return) / Variance(Benchmark_Return)

    Attributes:
        required_columns (Set[str]): Required columns for calculation.

    Example:
        ```python
        calculator = BetaCalculator()
        result = calculator.calculate(data)
        ```
    """

    type: MetricType = MetricType.POSITIVE

    def expression(self) -> pl.Expr:
        return (
            pl.cov(pl.col("Return"), pl.col("Benchmark_Return"))
            / pl.col("Benchmark_Return").var()
        )
