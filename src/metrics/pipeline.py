from typing import Dict, List, Optional


from .base import MetricCalculator, MetricType

import pandas as pd


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

    def get_calculator_types(self) -> Dict[str, MetricType]:
        """Get the types of all calculators in the pipeline.

        Returns:
            Dict[str, MetricType]: Dictionary mapping calculator names to their types.
        """
        return {i.__class__.__name__: i.type for i in self.calculators}

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the metric calculation pipeline.

        Args:
            data (pd.DataFrame): Input data for metric calculation.

        Returns:
            pd.DataFrame: Data with all calculated metrics added.
        """
        results = [metric.calculate_all_groups(data) for metric in self.calculators]

        # Merge results
        final_df = results[0]
        for df in results[1:]:
            final_df = final_df.merge(df, on=["PM_ID", "Strategy_ID"], how="outer")

        return final_df
