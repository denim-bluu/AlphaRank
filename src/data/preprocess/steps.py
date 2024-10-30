import pandas as pd

from src.data.preprocess.base import PreprocessingStep


class SortStep(PreprocessingStep):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(["PM_ID", "Strategy_ID", "Date"])


class OptimizeSchemaStep(PreprocessingStep):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.astype(
            {
                "Date": "datetime64[ns]",
                "Strategy_ID": "category",
                "Return": "float64",
                "Benchmark_Return": "float64",
            }
        )


class RollingMeanReturnStep:
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        rolling_mean_return = (
            df.groupby(["PM_ID", "Strategy_ID"], observed=False)["Return"]
            .rolling(window=12, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
            .rename("Rolling_12M_Return")
        )
        return df.assign(Rolling_12M_Return=rolling_mean_return.values)


class RollingMeanBenchmarkReturnStep:
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        rolling_mean_benchmark_return = (
            df.groupby(["PM_ID", "Strategy_ID"], observed=False)["Benchmark_Return"]
            .rolling(window=12, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
            .rename("Rolling_12M_Benchmark_Return")
        )
        return df.assign(
            Rolling_12M_Benchmark_Return=rolling_mean_benchmark_return.values
        )


class RollingStdReturnStep:
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        rolling_std_return = (
            df.groupby(["PM_ID", "Strategy_ID"], observed=False)["Return"]
            .rolling(window=12, min_periods=1)
            .std()
            .reset_index(level=[0, 1], drop=True)
            .rename("Rolling_12M_Volatility")
        )
        return df.assign(Rolling_12M_Volatility=rolling_std_return.values)


class RollingStdBenchmarkReturnStep:
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        rolling_std_benchmark_return = (
            df.groupby(["PM_ID", "Strategy_ID"], observed=False)["Benchmark_Return"]
            .rolling(window=12, min_periods=1)
            .std()
            .reset_index(level=[0, 1], drop=True)
            .rename("Rolling_12M_Benchmark_Volatility")
        )
        return df.assign(
            Rolling_12M_Benchmark_Volatility=rolling_std_benchmark_return.values
        )


class CumulativeReturnStep(PreprocessingStep):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        cumulative_return = df.groupby(["PM_ID", "Strategy_ID"], observed=False)[
            "Return"
        ].apply(lambda x: (1 + x).cumprod() - 1)
        cumulative_benchmark_return = df.groupby(
            ["PM_ID", "Strategy_ID"], observed=False
        )["Benchmark_Return"].apply(lambda x: (1 + x).cumprod() - 1)
        return df.assign(
            Cumulative_Return=cumulative_return.values,
            Cumulative_Benchmark_Return=cumulative_benchmark_return.values,
        )
