from abc import ABC, abstractmethod

import polars as pl


class DataSource(ABC):
    @abstractmethod
    def fetch_data(self) -> pl.LazyFrame:
        pass


class ParquetDataSource(DataSource):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def fetch_data(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.file_path)
