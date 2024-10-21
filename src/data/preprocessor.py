from abc import ABC, abstractmethod

import polars as pl


class PreprocessingStep(ABC):
    @abstractmethod
    def apply(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        pass


class SortStep(PreprocessingStep):
    def apply(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.sort(["PM_ID", "Strategy_ID", "Date"])


class RollingOperationStep(PreprocessingStep):
    def __init__(self, column: str, window_size: int, operation: str, alias: str):
        self.column = column
        self.window_size = window_size
        self.operation = operation
        self.alias = alias

    def apply(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        if self.operation == "mean":
            return lf.with_columns(
                pl.col(self.column)
                .rolling_mean(window_size=self.window_size)
                .over(["PM_ID", "Strategy_ID"])
                .alias(self.alias)
            )
        elif self.operation == "std":
            return lf.with_columns(
                pl.col(self.column)
                .rolling_std(window_size=self.window_size)
                .over(["PM_ID", "Strategy_ID"])
                .alias(self.alias)
            )
        else:
            raise ValueError(f"Unsupported operation: {self.operation}")


class RollingMeanReturnStep(RollingOperationStep):
    def __init__(self):
        super().__init__("Return", 12, "mean", "Rolling_12M_Return")


class RollingMeanBenchmarkReturnStep(RollingOperationStep):
    def __init__(self):
        super().__init__("Benchmark_Return", 12, "mean", "Rolling_12M_Benchmark_Return")


class RollingStdReturnStep(RollingOperationStep):
    def __init__(self):
        super().__init__("Return", 12, "std", "Rolling_12M_Volatility")


class RollingStdBenchmarkReturnStep(RollingOperationStep):
    def __init__(self):
        super().__init__(
            "Benchmark_Return", 12, "std", "Rolling_12M_Benchmark_Volatility"
        )


class DataPreprocessor:
    def __init__(self, steps: list[PreprocessingStep]):
        self.steps = steps

    def preprocess(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        for step in self.steps:
            lf = step.apply(lf)
        return lf
