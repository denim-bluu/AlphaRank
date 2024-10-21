from abc import ABC, abstractmethod

import polars as pl


class MetricStandardizer(ABC):
    """Abstract base class for metric standardizers."""

    @abstractmethod
    def standardize(
        self, data: pl.LazyFrame, metric_columns: list[str]
    ) -> pl.LazyFrame:
        """
        Standardize metric columns in the given data.

        Args:
            data (pl.LazyFrame): Input data containing the metrics to standardize.
            metric_columns (list[str]): Names of the columns to standardize.

        Returns:
            pl.LazyFrame: Data with the standardized metrics added.
        """
        pass


class ZScoreStandardizer(MetricStandardizer):
    """Standardizer that uses Z-Score normalization."""

    def standardize(
        self, data: pl.LazyFrame, metric_columns: list[str]
    ) -> pl.LazyFrame:
        for column in metric_columns:
            data = data.with_columns(
                [
                    (
                        (pl.col(column) - pl.col(column).mean()) / pl.col(column).std()
                    ).alias(f"{column}_standardized")
                ]
            )
        return data


class MinMaxStandardizer(MetricStandardizer):
    """Standardizer that uses Min-Max normalization."""

    def standardize(
        self, data: pl.LazyFrame, metric_columns: list[str]
    ) -> pl.LazyFrame:
        for column in metric_columns:
            data = data.with_columns(
                [
                    (
                        (pl.col(column) - pl.col(column).min())
                        / (pl.col(column).max() - pl.col(column).min())
                    ).alias(f"{column}_standardized")
                ]
            )
        return data


class StandardizerFactory:
    """Factory for creating standardizers."""

    @staticmethod
    def create_standardizer(standardizer_type: str) -> MetricStandardizer:
        """
        Create a standardizer instance.

        Args:
            standardizer_type (str): Type of standardizer to create ('zscore' or 'minmax').

        Returns:
            MetricStandardizer: Instance of the requested standardizer.

        Raises:
            ValueError: If the standardizer type is unknown.
        """
        if standardizer_type == "zscore":
            return ZScoreStandardizer()
        elif standardizer_type == "minmax":
            return MinMaxStandardizer()
        else:
            raise ValueError(f"Unknown standardizer type: {standardizer_type}")
