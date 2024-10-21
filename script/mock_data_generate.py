from datetime import datetime, timedelta

import numpy as np
import polars as pl

num_pms = 10
num_strategies_per_pm = 3
num_years = 5

# Generate date range
start_date = datetime(2018, 1, 1).date()
end_date = start_date + timedelta(days=365 * num_years)
dates = pl.date_range(start_date, end_date, interval="1mo", eager=True)

# Define regions and benchmarks
regions = ["North America", "Europe", "Asia", "Emerging Markets"]
benchmarks = {
    "North America": "S&P 500",
    "Europe": "STOXX Europe 600",
    "Asia": "MSCI AC Asia",
    "Emerging Markets": "MSCI Emerging Markets",
}

data = []
for pm_id in range(1, num_pms + 1):
    for strategy_id in range(1, num_strategies_per_pm + 1):
        region = np.random.choice(regions)
        benchmark = benchmarks[region]

        for date in dates:
            return_val = np.random.normal(
                0.005, 0.03
            )  # Mean 0.5% monthly return, 3% std dev
            benchmark_return = np.random.normal(
                0.004, 0.025
            )  # Slightly lower return and volatility for benchmark

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

df = pl.DataFrame(data)
df = df.sort(["PM_ID", "Strategy_ID", "Date"])
df = df.with_columns(
    ((1 + pl.col("Return")).cum_prod().over(["PM_ID", "Strategy_ID"]) - 1).alias(
        "Cumulative_Return"
    ),
    (
        (1 + pl.col("Benchmark_Return")).cum_prod().over(["PM_ID", "Strategy_ID"]) - 1
    ).alias("Cumulative_Benchmark_Return"),
)
df.write_parquet("mock_performance_data.parquet")

print("Mock data generated and saved to 'mock_performance_data.parquet'")
