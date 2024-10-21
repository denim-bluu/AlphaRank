from .base import MetricStandardizer
import polars as pl


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
