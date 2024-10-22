from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import polars as pl
from loguru import logger


class StrategyScoreAggregator(ABC):
    """Abstract base class for implementing score aggregation strategies.

    This class provides a template for implementing different score aggregation
    strategies. It handles common validation logic and defines the interface
    that all concrete aggregation strategies must implement.

    Attributes:
        name (str): Name of the aggregation strategy. Defaults to class name if not provided.

    Example:
        ```python
        class WeightedAverage(StrategyScoreAggregator):
            def _aggregate(self, data, metric_columns, weights):
                # Implementation here
                pass

        aggregator = WeightedAverage(name="CustomWeighted")
        result = aggregator.aggregate(data, ["metric1", "metric2"], {"metric1": 0.6, "metric2": 0.4})
        ```
    """

    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize the aggregator with an optional custom name.

        Args:
            name: Optional custom name for the aggregator. If not provided,
                uses the class name.
        """
        self.name = name or self.__class__.__name__

    def _check_weights(
        self, weights: Dict[str, float], metric_columns: List[str]
    ) -> None:
        """Validate the weights dictionary against required criteria.

        Performs validation checks on the weights dictionary to ensure:
        - Weights sum to 1 (within floating-point precision)
        - All weights are positive
        - Weights keys match metric columns exactly

        Args:
            weights: Dictionary mapping metric names to their weights.
            metric_columns: List of metric column names to validate against.

        Raises:
            ValueError: If any validation check fails:
                - Empty weights dictionary
                - Weights don't sum to 1
                - Negative weights
                - Mismatch between weights and metric columns
        """
        if not weights:
            logger.error("Weights dictionary cannot be empty")
            raise ValueError("Weights dictionary cannot be empty")

        if set(weights.keys()) != set(metric_columns):
            logger.error("Weights keys must match metric columns exactly")
            raise ValueError("Weights keys must match metric columns exactly")

        if not (
            0.99 <= sum(weights.values()) <= 1.01
        ):  # Allow for floating-point imprecision
            logger.error(f"Weights must sum up to 1, got {sum(weights.values())}")
            raise ValueError(f"Weights must sum up to 1, got {sum(weights.values())}")

        if any(weight < 0 for weight in weights.values()):
            logger.error("All weights must be positive")
            raise ValueError("All weights must be positive")

    def _validate_input(self, data: pl.LazyFrame, metric_columns: List[str]) -> None:
        """Validate input data and required metric columns.

        Checks if all required metric columns exist in the input DataFrame.

        Args:
            data: Input LazyFrame to validate.
            metric_columns: List of required metric column names.

        Raises:
            ValueError: If any required columns are missing from the input data.
        """
        schema = data.collect_schema()
        missing_cols = [col for col in metric_columns if col not in schema]
        if missing_cols:
            logger.error(f"Missing columns in data: {missing_cols}")
            raise ValueError(f"Missing columns in data: {missing_cols}")

    @abstractmethod
    def _aggregate(
        self, data: pl.LazyFrame, metric_columns: List[str], weights: Dict[str, float]
    ) -> pl.LazyFrame:
        """Abstract method to implement the specific aggregation strategy.

        This method must be implemented by concrete subclasses to define
        how metrics should be aggregated.

        Args:
            data: LazyFrame containing the data to aggregate.
            metric_columns: List of column names to include in aggregation.
            weights: Dictionary mapping metric names to their weights in the aggregation.

        Returns:
            LazyFrame with aggregated results added.

        Raises:
            NotImplementedError: If the concrete class doesn't implement this method.
        """
        raise NotImplementedError

    def aggregate(
        self, data: pl.LazyFrame, metric_columns: List[str], weights: Dict[str, float]
    ) -> pl.LazyFrame:
        """Aggregate multiple metrics into a single score using the defined strategy.

        This is the main public interface for the aggregator. It performs validation
        and then calls the specific aggregation implementation.

        Args:
            data: Input LazyFrame containing the metrics to aggregate.
            metric_columns: List of column names to include in the aggregation.
            weights: Dictionary mapping metric names to their weights in the aggregation.

        Returns:
            LazyFrame with the aggregated score added.

        Raises:
            ValueError: If input validation fails or weights are invalid.

        Example:
            ```python
            aggregator = WeightedAverageAggregator()
            weights = {"metric1": 0.6, "metric2": 0.4}
            result = aggregator.aggregate(data, ["metric1", "metric2"], weights)
            ```
        """
        logger.debug(f"Starting aggregation with strategy: {self.name}")

        self._validate_input(data, metric_columns)
        self._check_weights(weights, metric_columns)

        result = self._aggregate(data, metric_columns, weights)

        logger.debug(f"Completed aggregation with strategy: {self.name}")
        return result
