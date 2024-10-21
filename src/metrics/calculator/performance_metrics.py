import polars as pl

from .base import MetricCalculator


class ExcessReturnCalculator(MetricCalculator):
    """
    ExcessReturnCalculator calculates the excess return for a given dataset.

    Excess return is the return of an investment compared to a market index or benchmark.
    It represents the difference between the return of an investment and the return of a benchmark index.

    Formula:
        Excess Return = Return - Benchmark_Return

    Methods:
        calculate(data: pl.LazyFrame) -> pl.LazyFrame:
            Calculates the excess return for the given data.

    Args:
        data (pl.LazyFrame): A LazyFrame containing the columns "Return" and "Benchmark_Return".

    Returns:
        pl.LazyFrame: A LazyFrame with an additional column "Excess_Return" representing the calculated excess return values.
    """

    def calculate(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data.group_by("Strategy_ID").agg(
            [
                ((pl.col("Return") - pl.col("Benchmark_Return")).mean()).alias(
                    "Excess_Return"
                )
            ]
        )


class BetaCalculator(MetricCalculator):
    """
    BetaCalculator calculates the beta of a strategy relative to a benchmark.

    Beta is a measure of the volatility, or systematic risk, of a security or portfolio
    in comparison to the market as a whole. It is used in the capital asset pricing model (CAPM)
    to describe the relationship between the expected return of an asset and its risk relative
    to the market.

    The formula for beta is:
        Beta = Cov(Return, Benchmark_Return) / Var(Benchmark_Return)

    Methods:
        calculate(data: pl.LazyFrame) -> pl.LazyFrame:
            Calculates the beta for each strategy in the provided data.
            Args:
                data (pl.LazyFrame): A lazy frame containing the columns "Return" and "Benchmark_Return".
            Returns:
                pl.LazyFrame: A lazy frame with an additional column "Beta" representing the calculated beta values.
    """

    def calculate(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data.group_by("Strategy_ID").agg(
            [
                (
                    pl.cov(pl.col("Return"), (pl.col("Benchmark_Return")))
                    / pl.col("Benchmark_Return").var()
                ).alias("Beta")
            ]
        )
