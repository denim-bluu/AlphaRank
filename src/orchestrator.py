from typing import Dict, List

import polars as pl
from loguru import logger

from .metrics.portfolio_aggregator.base import PortfolioScoreAggregator
from .metrics.standardizer.base import MetricStandardizer
from .metrics.strategy_aggregator.base import StrategyScoreAggregator

from .data.pipeline import DataPipeline
from .metrics.calculator.pipeline import CalculationPipeline

logger.add("performance_analysis.log", rotation="500 MB", level="INFO")


class PerformanceAnalysisOrchestrator:
    """Orchestrator for the entire performance analysis process."""

    def __init__(
        self,
        data_pipeline: DataPipeline,
        calculation_pipeline: CalculationPipeline,
        standardizer: MetricStandardizer,
        strategy_aggregator: StrategyScoreAggregator,
        portfolio_aggregator: PortfolioScoreAggregator,
    ):
        self._data_pipeline = data_pipeline
        self._calculation_pipeline = calculation_pipeline
        self._standardizer = standardizer
        self._strategy_aggregator = strategy_aggregator
        self._portfolio_aggregator = portfolio_aggregator

        self.metric_results: pl.LazyFrame | None = None
        self.scored_results: pl.LazyFrame | None = None

    def run_analysis(
        self, metric_columns: List[str], weights: Dict[str, float]
    ) -> bool:
        """
        Run the entire performance analysis process.

        Args:
            metric_columns (List[str]): Names of the metric columns to use in scoring.
            weights (Dict[str, float]): Weights for each metric in the aggregation.

        Returns:
            bool: True if the analysis was successful, False otherwise.
        """
        try:
            logger.info("Starting performance analysis")

            # Process data
            logger.info("Processing raw data")
            processed_data = self._data_pipeline.run()

            # Calculate metrics
            logger.info("Calculating performance metrics")
            metric_results = self._calculation_pipeline.run(processed_data)

            # Standardize metrics
            logger.info("Standardizing metrics")
            metric_results = self._standardizer.standardize(
                metric_results, metric_columns
            )

            # Aggregate strategy scores
            logger.info("Aggregating strategy scores")
            self.metric_results = self._strategy_aggregator.aggregate(
                metric_results, metric_columns, weights
            )

            # Aggregate portfolio scores
            logger.info("Aggregating portfolio scores")
            scored_results = self._portfolio_aggregator.aggregate(self.metric_results)

            # Add rankings
            self.scored_results = scored_results.with_columns(
                [pl.col("PM_Score").rank(method="dense", descending=True).alias("Rank")]
            ).sort("Rank")

            logger.success("✅ Performance analysis completed successfully")
            return True

        except Exception as e:
            logger.error(f"An error occurred during performance analysis: {str(e)}")
            raise e

    def get_metric_results(self) -> pl.DataFrame:
        """Get the calculated metric results."""
        if self.metric_results is None:
            raise ValueError("No metric results available. Run analysis first.")
        return self.metric_results.collect()

    def get_scored_results(self) -> pl.DataFrame:
        """Get the scored results with rankings."""
        if self.scored_results is None:
            raise ValueError("No scored results available. Run analysis first.")
        return self.scored_results.collect()
