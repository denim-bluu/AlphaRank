from abc import ABC, abstractmethod
from typing import Dict, List
import polars as pl


class WeightingMethod(ABC):
    """Base class for weighting methods"""

    @abstractmethod
    def calculate_weights(
        self, metric_data: pl.LazyFrame, metric_columns: List[str]
    ) -> Dict[str, float]:
        """Calculate weights for metrics"""
        pass
