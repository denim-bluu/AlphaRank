import polars as pl

from .base import MetricStandardizer


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
