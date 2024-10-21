import polars as pl
from polars.testing import assert_frame_equal

from src.metrics.calculator.performance_metrics import (
    BetaCalculator,
    ExcessReturnCalculator,
)


def test_excess_return_calculator():
    # Sample data
    data = pl.LazyFrame(
        {
            "Strategy_ID": [1, 1, 2, 2],
            "Return": [0.05, 0.07, 0.10, 0.12],
            "Benchmark_Return": [0.03, 0.04, 0.08, 0.09],
        }
    )

    # Expected result
    expected_data = pl.DataFrame(
        {"Strategy_ID": [1, 2], "Excess_Return": [0.025, 0.025]}
    )

    # Instantiate and calculate
    calculator = ExcessReturnCalculator()
    result = calculator.calculate(data).collect()
    # Assert the result
    assert_frame_equal(
        result.sort("Strategy_ID"),
        expected_data.sort("Strategy_ID"),
    )


def test_beta_calculator():
    # Sample data
    data = pl.LazyFrame(
        {
            "Strategy_ID": [1, 1, 2, 2],
            "Return": [0.06, 0.08, 0.16, 0.18],
            "Benchmark_Return": [0.03, 0.04, 0.08, 0.09],
        }
    )

    # Expected result
    expected_data = pl.DataFrame(
        {
            "Strategy_ID": [1, 2],
            "Beta": [
                2.0,
                2.0,
            ],
        }
    )

    # Instantiate and calculate
    calculator = BetaCalculator()
    result = calculator.calculate(data).collect()
    assert_frame_equal(
        result.sort("Strategy_ID"),
        expected_data.sort("Strategy_ID"),
    )
