from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    @abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        pass


class ParquetDataSource(DataSource):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def fetch_data(self) -> pd.DataFrame:
        return pd.read_parquet(self.file_path)
