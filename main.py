from src.data import preprocessor as pp
from src.data.pipeline import DataPipeline
from src.data.source import ParquetDataSource
from src.data.validator import DataValidator
from src.metrics.strategy_aggregator.strategy import StrategyScoreAggregator
from src.metrics.portfolio_aggregator.portfolio import PMScoreAggregator
from src.metrics.calculator.factory import MetricCalculatorFactory
from src.metrics.calculator.pipeline import CalculationPipeline
from src.metrics.scoring_pipeline import ScoringPipeline
from src.metrics.standardizer.factory import StandardizerFactory
from src.orchestrator import PerformanceAnalysisOrchestrator

data_source = ParquetDataSource("data/mock_performance_data.parquet")
validator = DataValidator()


preprocessor = pp.DataPreprocessor(pp.PreprocessorStepFactory.create_all_steps())

data_pipeline = DataPipeline(data_source, validator, preprocessor)

metric_pipeline = CalculationPipeline(MetricCalculatorFactory.create_all_calculators())

standardizer = StandardizerFactory.create_standardizer("zscore")
strategy_aggregator = StrategyScoreAggregator()
portfolio_aggregator = PMScoreAggregator()

scoring_pipeline = ScoringPipeline(
    standardizer, strategy_aggregator, portfolio_aggregator
)

metric_columns = ["Excess_Return", "Sharpe_Ratio", "Information_Ratio"]
weights = {"Excess_Return": 0.4, "Sharpe_Ratio": 0.3, "Information_Ratio": 0.3}

orchestrator = PerformanceAnalysisOrchestrator(
    data_pipeline=data_pipeline,
    calculation_pipeline=metric_pipeline,
    scoring_pipeline=scoring_pipeline,
)

result = orchestrator.run_analysis(metric_columns, weights)
