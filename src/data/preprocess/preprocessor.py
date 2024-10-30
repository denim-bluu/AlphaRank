import pandas as pd
from src.data.preprocess.base import PreprocessingStep


class DataPreprocessor:
    def __init__(self, steps: list[PreprocessingStep]):
        self.steps = steps

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            df = step.apply(df)
        return df
