from typing import Dict
import polars as pl

from src.calculators.base import MetricType

from src.standardizers.base import MetricStandardizer


class MinMaxStandardizer(MetricStandardizer):
    """Standardizer that uses Min-Max normalization."""

    def _positive_standardize(self, column: str) -> pl.Expr:
        return (pl.col(column) - pl.col(column).min()) / (
            pl.col(column).max() - pl.col(column).min()
        )

    def _negative_standardize(self, column: str) -> pl.Expr:
        return (pl.col(column).max() - pl.col(column)) / (
            pl.col(column).max() - pl.col(column).min()
        )

    def standardize(
        self, metric_data: pl.LazyFrame, metric_types: Dict[str, MetricType]
    ) -> pl.LazyFrame:
        for metric, metric_type in metric_types.items():
            if metric_type == MetricType.POSITIVE:
                metric_data = metric_data.with_columns(
                    self._positive_standardize(metric).alias(metric)
                )
            elif metric_type == MetricType.NEGATIVE:
                metric_data = metric_data.with_columns(
                    self._negative_standardize(metric).alias(metric)
                )
            else:
                raise ValueError(f"Invalid metric type: {metric_type}")
        return metric_data
