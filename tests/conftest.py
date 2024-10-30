from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

NUM_YEARS = 5
NUM_PMS = 10
NUM_STRATEGIES_PER_PM = 3


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Fixture to create a sample dataset for testing."""
    start_date = datetime(2018, 1, 1).date()
    end_date = start_date + timedelta(days=365 * NUM_YEARS)
    dates = pd.date_range(start_date, end_date, freq="ME")
    # Define regions and benchmarks
    regions = ["North America", "Europe", "Asia", "Emerging Markets"]
    benchmarks = {
        "North America": "S&P 500",
        "Europe": "STOXX Europe 600",
        "Asia": "MSCI AC Asia",
        "Emerging Markets": "MSCI Emerging Markets",
    }
    data = []
    for pm_id in range(1, NUM_PMS + 1):
        for strategy_id in range(1, NUM_STRATEGIES_PER_PM + 1):
            return_val = np.random.normal(
                0.005, 0.03
            )  # Mean 0.5% monthly return, 3% std dev
            benchmark_return = np.random.normal(
                0.004, 0.025
            )  # Slightly lower return and volatility for benchmark

            region = np.random.choice(regions)
            benchmark = benchmarks[region]

            for date in dates:
                data.append(
                    {
                        "PM_ID": f"PM_{pm_id:03d}",
                        "Strategy_ID": f"S_{pm_id:03d}_{strategy_id:03d}",
                        "Benchmark_ID": benchmark,
                        "Date": date,
                        "Region": region,
                        "Return": return_val,
                        "Benchmark_Return": benchmark_return,
                        "Excess_Return": return_val - benchmark_return,
                        "AUM": np.random.uniform(1e6, 1e9),
                    }
                )

    df = pd.DataFrame(data)
    df = df.sort_values(by=["PM_ID", "Strategy_ID", "Date"])
    cumulative_return = df.groupby(["PM_ID", "Strategy_ID"], observed=False)[
        "Return"
    ].apply(lambda x: (1 + x).cumprod() - 1)
    cumulative_benchmark_return = df.groupby(["PM_ID", "Strategy_ID"], observed=False)[
        "Benchmark_Return"
    ].apply(lambda x: (1 + x).cumprod() - 1)
    return df.assign(
        Cumulative_Return=cumulative_return.values,
        Cumulative_Benchmark_Return=cumulative_benchmark_return.values,
    )
