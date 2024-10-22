from abc import ABC, abstractmethod
from typing import List, Optional, Set

import polars as pl
from loguru import logger


class MetricCalculator(ABC):
    """Abstract base class for implementing financial metric calculations.

    This class provides a template for implementing different financial metric
    calculations. It handles common validation logic and defines the interface
    that all concrete metric calculators must implement.

    Attributes:
        name (str): Name of the metric calculator. Defaults to class name if not provided.
        required_columns (Set[str]): Set of column names required for the calculation.

    Example:
        ```python
        class CustomMetric(MetricCalculator):
            def __init__(self):
                super().__init__(required_columns={"Return", "Benchmark_Return"})

            def _calculate_metric(self, data):
                # Implementation here
                pass

        calculator = CustomMetric(name="CustomMetric")
        result = calculator.calculate(data)
        ```
    """

    def __init__(
        self,
        required_columns: Set[str],
        name: Optional[str] = None,
        group_by_columns: Optional[List[str]] = None,
    ) -> None:
        """Initialize the metric calculator.

        Args:
            required_columns: Set of column names required for the calculation.
            name: Optional custom name for the calculator.
            group_by_columns: Optional list of columns to group by during calculation.
                Defaults to ["PM_ID", "Strategy_ID"].
        """
        self.name = name or self.__class__.__name__
        self.required_columns = required_columns
        self.group_by_columns = group_by_columns or ["PM_ID", "Strategy_ID"]

    def _validate_input(self, data: pl.LazyFrame) -> None:
        """Validate input data against required columns.

        Args:
            data: Input LazyFrame to validate.

        Raises:
            ValueError: If any required columns are missing from the input data.
        """
        schema = data.collect_schema()
        missing_cols = [col for col in self.required_columns if col not in schema]
        if missing_cols:
            logger.error(f"Missing required columns in data: {missing_cols}")
            raise ValueError(f"Missing required columns in data: {missing_cols}")

        missing_group_cols = [col for col in self.group_by_columns if col not in schema]
        if missing_group_cols:
            logger.error(f"Missing grouping columns in data: {missing_group_cols}")
            raise ValueError(f"Missing grouping columns in data: {missing_group_cols}")

    @abstractmethod
    def _calculate_metric(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """Abstract method to implement the specific metric calculation.

        This method must be implemented by concrete subclasses to define
        how the metric should be calculated.

        Args:
            data: LazyFrame containing the required data for calculation.

        Returns:
            LazyFrame with calculated metric added.

        Raises:
            NotImplementedError: If the concrete class doesn't implement this method.
        """
        raise NotImplementedError

    def calculate(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """Calculate the metric for the provided data.

        This is the main public interface for the calculator. It performs validation
        and then calls the specific metric calculation implementation.

        Args:
            data: Input LazyFrame containing required data for calculation.

        Returns:
            LazyFrame with the calculated metric added.

        Raises:
            ValueError: If input validation fails.
        """
        logger.debug(f"Starting metric calculation with {self.name}")

        self._validate_input(data)
        result = self._calculate_metric(data)

        logger.debug(f"Completed metric calculation with {self.name}")
        return result
