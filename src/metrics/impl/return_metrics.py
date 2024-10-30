from typing import Tuple
import numpy as np
import pandas as pd

from src.metrics.base import MetricCalculator, MetricType


class ExcessReturn(MetricCalculator):
    """Calculator for excess return metric.

    Calculates the excess return, which is the difference between the return
    of an investment and the return of a benchmark index.

    Formula:
        Excess Return = Average(Return - Benchmark_Return)

    Example:
        ```python
        calculator = ExcessReturnCalculator()
        result = calculator.calculate(data)
        ```
    """

    type: MetricType = MetricType.POSITIVE

    def calculate_for_group(self, group_data: pd.DataFrame) -> float:
        return (group_data["Return"] - group_data["Benchmark_Return"]).mean()


class Beta(MetricCalculator):
    """Calculator for beta metric.

    Calculates beta, which measures the volatility of a security or portfolio
    relative to the market. Used in CAPM to describe the relationship between
    expected return and systematic risk.

    Formula:
        Beta = Covariance(Return, Benchmark_Return) / Variance(Benchmark_Return)

    Example:
        ```python
        calculator = BetaCalculator()
        result = calculator.calculate(data)
        ```
    """

    type: MetricType = MetricType.POSITIVE

    def calculate_for_group(self, group_data: pd.DataFrame) -> float:
        return (
            group_data[["Return", "Benchmark_Return"]].cov().values[0, 1]
            / group_data["Benchmark_Return"].var()
        )


class JensensAlpha(MetricCalculator):
    """Calculates Jensen's Alpha which measures the excess return of a portfolio
    over what would be predicted by CAPM given the portfolio's beta.

    https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1968.tb00815.x

    Formula:
        alpha = Rp - [Rf + beta(Rm - Rf)]
        where:
        Rp = Portfolio return
        Rf = Risk-free rate
        Rm = Market (benchmark) return
        beta = Portfolio beta
    """

    type = MetricType.POSITIVE

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        self.monthly_rf = risk_free_rate / 12

    def calculate_for_group(self, group_data: pd.DataFrame) -> float:
        # Calculate excess returns
        excess_portfolio = group_data["Return"] - self.monthly_rf
        excess_market = group_data["Benchmark_Return"] - self.monthly_rf

        # Calculate beta using covariance method
        beta = np.cov(excess_portfolio, excess_market)[0, 1] / np.var(excess_market)

        # Calculate average returns
        avg_portfolio_return = group_data["Return"].mean()
        avg_market_return = group_data["Benchmark_Return"].mean()

        # Calculate Jensen's Alpha
        return avg_portfolio_return - (
            self.monthly_rf + beta * (avg_market_return - self.monthly_rf)
        )


class TreynorMazuyMeasure(MetricCalculator):
    """
    Calculates Treynor-Mazuy measure which evaluates market timing ability.

    Formula:
        Rp - Rf = alpha + beta(Rm - Rf) + gamma(Rm - Rf)^2 + epsilon
        where:
        Rp = Portfolio return
        Rf = Risk-free rate
        Rm = Market return
        alpha = selectivity
        beta = market sensitivity
        gamma = market timing ability
    """

    type = MetricType.POSITIVE

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        self.monthly_rf = risk_free_rate / 12

    def calculate_for_group(
        self, group_data: pd.DataFrame
    ) -> Tuple[float, float, float]:
        # Calculate excess returns
        excess_portfolio = group_data["Return"] - self.monthly_rf
        excess_market = group_data["Benchmark_Return"] - self.monthly_rf

        # Prepare data for quadratic regression
        _x = np.column_stack(
            [np.ones_like(excess_market), excess_market, excess_market**2]
        )

        # Perform quadratic regression
        coefficients = np.linalg.lstsq(_x, excess_portfolio, rcond=None)[0]

        # Extract coefficients
        _ = coefficients[0]  # Selectivity, alpha
        _ = coefficients[1]  # Market sensitivity, beta
        gamma = coefficients[2]  # Market timing ability

        return gamma
