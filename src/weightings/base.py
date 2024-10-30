from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd


class WeightingMethod(ABC):
    """Base class for weighting methods"""

    @abstractmethod
    def calculate_weights(
        self, metric_data: pd.DataFrame, metric_columns: List[str]
    ) -> Dict[str, float]:
        """Calculate weights for metrics"""
        pass
