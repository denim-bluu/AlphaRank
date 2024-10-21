import numpy as np
import polars as pl

from .base import MetricCalculator


class SharpeRatioCalculator(MetricCalculator):
    """
    A calculator for the Sharpe Ratio metric.

    The Sharpe Ratio is a measure of risk-adjusted return. It is calculated as the
    difference between the return of an investment and the risk-free rate, divided
    by the standard deviation of the investment's return. The formula is:

        Sharpe Ratio = (Mean(Return) - Risk-Free Rate) / Std(Return) * sqrt(12)

    Attributes:
        risk_free_rate (float): The annual risk-free rate, default is 0.02 (2%).

    Methods:
        calculate(data: pl.LazyFrame) -> pl.LazyFrame:
            Calculates the Sharpe Ratio for each strategy in the provided data.
            The data should contain a column named "Return" and a column named
            "Strategy_ID".
    """

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate / 12  # Monthly risk-free rate

    def calculate(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data.group_by("Strategy_ID").agg(
            [
                (
                    (pl.col("Return").mean() - self.risk_free_rate)
                    / pl.col("Return").std()
                    * np.sqrt(12)
                ).alias("Sharpe_Ratio")
            ]
        )


class InformationRatioCalculator(MetricCalculator):
    """
    A calculator for the Information Ratio metric.

    The Information Ratio (IR) measures the risk-adjusted return of a financial asset or portfolio relative to a benchmark. It is calculated as the difference between the mean return of the asset and the mean return of the benchmark, divided by the standard deviation of the difference between the asset return and the benchmark return, scaled by the square root of 12 (to annualize the metric).

    Formula:
        IR = (mean(Return) - mean(Benchmark_Return)) / (std(Return - Benchmark_Return) * sqrt(12))

    Methods:
        calculate(data: pl.LazyFrame) -> pl.LazyFrame:
            Calculates the Information Ratio for each strategy in the provided data.

    Args:
        data (pl.LazyFrame): A Polars LazyFrame containing the columns "Return", "Benchmark_Return", and "Strategy_ID".

    Returns:
        pl.LazyFrame: A Polars LazyFrame with an additional column "Information_Ratio" containing the calculated Information Ratio for each strategy.
    """

    def calculate(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data.group_by("Strategy_ID").agg(
            [
                (
                    (pl.col("Return").mean() - pl.col("Benchmark_Return").mean())
                    / (pl.col("Return") - pl.col("Benchmark_Return")).std()
                    * np.sqrt(12)
                ).alias("Information_Ratio")
            ]
        )
