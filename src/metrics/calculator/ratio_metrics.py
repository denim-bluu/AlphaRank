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


class SortinoRatioCalculator(MetricCalculator):
    """
    A calculator for the Sortino Ratio metric.

    The Sortino Ratio is a measure of risk-adjusted return that focuses on the downside risk of an investment. It is calculated as the difference between the return of an investment and the risk-free rate, divided by the standard deviation of the investment's negative returns. The formula is:

    Formula:
        Sortino Ratio = (Mean(Return) - Risk-Free Rate) / Std(Negative_Return) * sqrt(12)
    """

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate / 12  # Monthly risk-free rate

    def calculate(self, data: pl.LazyFrame) -> pl.LazyFrame:
        negative_returns = pl.when(pl.col("Return") < 0).then(pl.col("Return"))
        return data.group_by("Strategy_ID").agg(
            [
                (
                    (pl.col("Return").mean() - self.risk_free_rate)
                    / negative_returns.std()
                    * np.sqrt(12)
                ).alias("Sortino_Ratio")
            ]
        )


class OmegaRatioCalculator(MetricCalculator):
    """
    A calculator for the Omega Ratio metric.

    The Omega Ratio is a risk-adjusted performance measure that evaluates the probability-weighted return distribution of an investment. It is calculated as the ratio of the expected gains to the expected losses, where gains are defined as returns above a specified threshold and losses are returns below the threshold.

    Formula:
        Omega Ratio = sum(Gains) / sum(Losses)
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def calculate(self, data: pl.LazyFrame) -> pl.LazyFrame:
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
            .group_by("Strategy_ID")
            .agg(
                [(pl.col("Gains").sum() / pl.col("Losses").sum()).alias("Omega_Ratio")]
            )
        )
