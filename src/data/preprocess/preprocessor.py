from src.data.preprocess.base import PreprocessingStep


import polars as pl


class DataPreprocessor:
    def __init__(self, steps: list[PreprocessingStep]):
        self.steps = steps

    def preprocess(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        for step in self.steps:
            lf = step.apply(lf)
        return lf
