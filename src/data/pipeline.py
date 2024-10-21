import polars as pl

from .preprocessor import DataPreprocessor
from .source import DataSource
from .validator import DataValidator


class DataPipeline:
    def __init__(
        self,
        data_source: DataSource,
        validator: DataValidator,
        preprocessor: DataPreprocessor,
    ):
        self.data_source = data_source
        self.validator = validator
        self.preprocessor = preprocessor

    def run(self) -> pl.LazyFrame:
        data = self.data_source.fetch_data()
        data = self.validator.validate(data)
        data = self.preprocessor.preprocess(data)
        return data
