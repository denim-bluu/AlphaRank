from dataclasses import dataclass
from typing import Dict, List
from src.weightings.base import WeightingMethod

import numpy as np
import pandas as pd


@dataclass
class EntropyWeightingMetadata:
    """Metadata for entropy-based weighting"""

    entropy_values: Dict[str, float]
    weights: Dict[str, float]
    diversity_degree: float  # Overall diversity of information


class EntropyWeighting(WeightingMethod):
    def _calculate_entropy(
        self, metric_data: pd.DataFrame, metric_columns: List[str]
    ) -> Dict[str, float]:
        """
        Calculate entropy for each metric
        """
        entropies = {}
        for col in metric_columns:
            values = metric_data[col].to_numpy()

            probs = values / values.sum()
            probs = np.where(probs == 0, 1e-10, probs)

            # Calculate entropy
            entropy = -np.sum(probs * np.log(probs)) / np.log(len(probs))
            entropies[col] = entropy

        return entropies

    def _calculate_weights(self, entropies: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate weights based on entropy values
        """
        diversities = {metric: 1 - entropy for metric, entropy in entropies.items()}
        total_diversity = sum(diversities.values())
        weights = {metric: div / total_diversity for metric, div in diversities.items()}
        return weights

    def calculate_weights(
        self, metric_data: pd.DataFrame, metric_columns: List[str]
    ) -> Dict[str, float]:
        """
        Calculate entropy-based weights for given metrics

        Args:
            metrics: List of metrics to consider

        Returns:
            Dictionary of metric weights
        """
        entropies = self._calculate_entropy(metric_data, metric_columns)
        weights = self._calculate_weights(entropies)
        self.metadata = EntropyWeightingMetadata(
            entropy_values=entropies,
            weights=weights,
            diversity_degree=1 - sum(entropies.values()) / len(entropies),
        )
        return weights
