from typing import Dict

from src.metrics.base import MetricType

from src.standardizers.base import MetricStandardizer
import pandas as pd


class MinMaxStandardizer(MetricStandardizer):
    """Standardizer that uses Min-Max normalization."""

    def standardize(
        self, metric_data: pd.DataFrame, metric_types: Dict[str, MetricType]
    ) -> pd.DataFrame:
        for metric, metric_type in metric_types.items():
            if metric_type == MetricType.POSITIVE:
                metric_data[metric] = (
                    metric_data[metric] - metric_data[metric].min()
                ) / (metric_data[metric].max() - metric_data[metric].min())
            elif metric_type == MetricType.NEGATIVE:
                metric_data[metric] = (
                    metric_data[metric].max() - metric_data[metric]
                ) / (metric_data[metric].max() - metric_data[metric].min())
            else:
                raise ValueError(f"Invalid metric type: {metric_type}")
        return metric_data
