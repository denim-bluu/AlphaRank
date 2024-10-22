from typing import List

import polars as pl
from loguru import logger

from .base import MetricCalculator


class CalculationPipeline:
    """Pipeline for calculating multiple metrics."""

    def __init__(self, calculators: List[MetricCalculator]):
        self.calculators = calculators

    def run(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Run the metric calculation pipeline.

        Args:
            data (pl.LazyFrame): Input data for metric calculation.

        Returns:
            pl.LazyFrame: Data with all calculated metrics added.
        """
        results: List[pl.LazyFrame] = []
        for calculator in self.calculators:
            try:
                logger.info(f"Calculating metric: {calculator.__class__.__name__}")
                result = calculator.calculate(data)
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Error calculating {calculator.__class__.__name__}: {str(e)}"
                )

        if not results:
            logger.critical("No metrics were successfully calculated")
            raise ValueError("No metrics were successfully calculated")

        # Combine all results
        combined_results = results[0]
        for result in results[1:]:
            combined_results = combined_results.join(
                result, on=["PM_ID", "Strategy_ID"], how="outer"
            )
            # Drop the duplicate Strategy_ID column
            combined_results = combined_results.drop(
                ["PM_ID_right", "Strategy_ID_right"]
            )

        return combined_results
