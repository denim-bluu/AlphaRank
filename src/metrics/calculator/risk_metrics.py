import numpy as np
import polars as pl

from .base import MetricCalculator


class VolatilityCalculator(MetricCalculator):
    """
    VolatilityCalculator calculates the volatility of returns for different strategies.

    Volatility is a statistical measure of the dispersion of returns for a given security or market index.
    It is often measured using the standard deviation or variance between returns from that same security or market index.

    Methods:
        calculate(data: pl.LazyFrame) -> pl.LazyFrame:
            Calculates the volatility of returns for each strategy.

    Args:
        data (pl.LazyFrame): A LazyFrame containing the return data with a column named "Return" and a column named "Strategy_ID".

    Returns:
        pl.LazyFrame: A LazyFrame with an additional column "Volatility" representing the calculated volatility for each strategy.
    """

    def calculate(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data.group_by("Strategy_ID").agg(
            [pl.col("Return").std().alias("Volatility")]
        )


class TrackingErrorCalculator(MetricCalculator):
    """Calculates the tracking error of a strategy's returns against a benchmark.

    Tracking error measures the standard deviation of the difference between the
    returns of a strategy and its benchmark. It is used to gauge how closely a
    portfolio follows its benchmark. A lower tracking error indicates that the
    portfolio returns are closely aligned with the benchmark returns.

    Formula:
        Tracking Error = sqrt(12) * std(Return - Benchmark_Return)

    Methods:
        calculate(data: pl.LazyFrame) -> pl.LazyFrame:
            Calculates the tracking error for each strategy in the provided data.
            Args:
                data (pl.LazyFrame): A LazyFrame containing the columns "Return", "Benchmark_Return", and "Strategy_ID".
            Returns:
                pl.LazyFrame: A LazyFrame with an additional column "Tracking_Error" representing the calculated tracking error for each strategy.
    """

    def calculate(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data.group_by("Strategy_ID").agg(
            [
                (
                    (pl.col("Return") - pl.col("Benchmark_Return")).std() * np.sqrt(12)
                ).alias("Tracking_Error")
            ]
        )
