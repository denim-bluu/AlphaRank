from typing import Dict, List

import polars as pl
from loguru import logger

from .data.pipeline import DataPipeline
from .metrics.calculator.pipeline import MetricCalculationPipeline
from .metrics.scoring_pipeline import ScoringPipeline

# Configure Loguru
logger.add("performance_analysis.log", rotation="500 MB", level="INFO")


class PerformanceAnalysisOrchestrator:
    """Orchestrator for the entire performance analysis process."""

    def __init__(
        self,
        data_pipeline: DataPipeline,
        metric_pipeline: MetricCalculationPipeline,
        scoring_pipeline: ScoringPipeline,
    ):
        self.data_pipeline = data_pipeline
        self.metric_pipeline = metric_pipeline
        self.scoring_pipeline = scoring_pipeline

    def run_analysis(
        self, metric_columns: List[str], weights: Dict[str, float]
    ) -> pl.DataFrame:
        """
        Run the entire performance analysis process.

        Args:
            metric_columns (List[str]): Names of the metric columns to use in scoring.
            weights (Dict[str, float]): Weights for each metric in the aggregation.

        Returns:
            pl.DataFrame: Final results with calculated metrics, scores, and rankings.
        """
        try:
            logger.info("Starting performance analysis")

            # Process data
            logger.info("Processing raw data")
            processed_data = self.data_pipeline.run()

            # Calculate metrics
            logger.info("Calculating performance metrics")
            metric_results = self.metric_pipeline.run(processed_data)

            # Score and rank strategies
            logger.info("Scoring and ranking strategies")
            scored_results = self.scoring_pipeline.run(
                metric_results, metric_columns, weights
            )

            # Add rankings
            final_results = scored_results.with_columns(
                [
                    pl.col("aggregated_score")
                    .rank(method="dense", descending=True)
                    .alias("rank")
                ]
            )

            logger.info("Performance analysis completed successfully")
            return final_results.collect()  # Materialize the LazyFrame

        except Exception as e:
            logger.error(f"An error occurred during performance analysis: {str(e)}")
            raise