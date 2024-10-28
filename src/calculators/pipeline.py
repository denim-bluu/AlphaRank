from typing import Dict, List, Optional

import polars as pl
from loguru import logger

from .base import MetricCalculator, MetricType


class CalculationPipeline:
    """Pipeline for calculating multiple metrics."""

    def __init__(
        self,
        calculators: List[MetricCalculator],
        group_by_columns: Optional[List[str]] = None,
    ) -> None:
        """Initialize the metric calculator.

        Args:
            calculators (List[MetricCalculator]): List of metric calculators to run.
            group_by_columns (Optional[List[str]], optional): Columns to group the data by. Defaults to None.
        """
        self.calculators = calculators
        self.group_by_columns = group_by_columns or ["PM_ID", "Strategy_ID"]

    def _validate_input(self, data: pl.LazyFrame) -> None:
        """Validate input data against required columns.

        Args:
            data: Input LazyFrame to validate.

        Raises:
            ValueError: If any required columns are missing from the input data.
        """
        schema = data.collect_schema()
        missing_group_cols = [col for col in self.group_by_columns if col not in schema]
        if missing_group_cols:
            logger.error(f"Missing grouping columns in data: {missing_group_cols}")
            raise ValueError(f"Missing grouping columns in data: {missing_group_cols}")

    def get_calculator_types(self) -> Dict[str, MetricType]:
        """Get the types of all calculators in the pipeline.

        Returns:
            Dict[str, MetricType]: Dictionary mapping calculator names to their types.
        """
        return {i.__class__.__name__: i.type for i in self.calculators}

    def run(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Run the metric calculation pipeline.

        Args:
            data (pl.LazyFrame): Input data for metric calculation.

        Returns:
            pl.LazyFrame: Data with all calculated metrics added.
        """
        exprs = [i.expression().alias(i.__class__.__name__) for i in self.calculators]
        self._validate_input(data)
        return data.group_by(self.group_by_columns).agg(exprs)
