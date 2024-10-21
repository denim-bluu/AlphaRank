import numpy as np
import polars as pl
import pytest

from src.metrics.calculator.ratio_metrics import InformationRatioCalculator


def test_information_ratio_calculator(sample_data):
    sample_data = pl.LazyFrame(
        {
            "Strategy_ID": ["A", "A", "B", "B"],
            "Return": [0.05, 0.10, 0.02, 0.03],
            "Benchmark_Return": [0.03, 0.04, 0.04, 0.02],
        }
    )
    calculator = InformationRatioCalculator()
    result = calculator.calculate(sample_data).collect()

    assert isinstance(result, pl.DataFrame)
    assert "Strategy_ID" in result.columns
    assert "Information_Ratio" in result.columns

    # Verify the calculated Information Ratio values
    expected_ratios = {
        "A": (0.04) / 0.028284 * np.sqrt(12),
        "B": (-0.005) / 0.021213 * np.sqrt(12),
    }
    for row in result.iter_rows():
        strategy_id = row[0]
        calculated_ratio = row[1]
        assert pytest.approx(calculated_ratio, rel=1e-2) == expected_ratios[strategy_id]
