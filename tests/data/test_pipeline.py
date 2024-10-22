import polars as pl

from src.data import preprocessor as pp
from src.data.pipeline import DataPipeline
from src.data.source import DataSource
from src.data.validator import DataValidator


class MockDataSource(DataSource):
    def __init__(self, data):
        self.data = data

    def fetch_data(self):
        return self.data


def test_data_pipeline(sample_data):
    mock_source = MockDataSource(sample_data)
    validator = DataValidator()
    steps = [
        pp.SortStep(),
        pp.RollingMeanReturnStep(),
        pp.RollingMeanBenchmarkReturnStep(),
        pp.RollingStdReturnStep(),
        pp.RollingStdBenchmarkReturnStep(),
    ]
    preprocessor = pp.DataPreprocessor(steps)

    pipeline = DataPipeline(mock_source, validator, preprocessor)
    result = pipeline.run()
    columns = result.collect_schema()
    assert isinstance(result, pl.LazyFrame)
    assert "Strategy_ID" in columns
    assert "Date" in columns
    assert "Return" in columns
    assert "Benchmark_Return" in columns
    assert "AUM" in columns
