from src.data import preprocessor as pp
from src.data.pipeline import DataPipeline
from src.data.source import ParquetDataSource
from src.data.validator import DataValidator
from src.metrics.aggregator.weighted_sum import WeightedSumAggregator
from src.metrics.calculator.factory import MetricCalculatorFactory
from src.metrics.calculator.pipeline import MetricCalculationPipeline
from src.metrics.scoring_pipeline import ScoringPipeline
from src.metrics.standardizer.factory import StandardizerFactory
from src.orchestrator import PerformanceAnalysisOrchestrator

data_source = ParquetDataSource("data/mock_performance_data.parquet")
validator = DataValidator()

steps = [
    pp.SortStep(),
    pp.RollingMeanReturnStep(),
    pp.RollingMeanBenchmarkReturnStep(),
    pp.RollingStdReturnStep(),
    pp.RollingStdBenchmarkReturnStep(),
]

preprocessor = pp.DataPreprocessor(steps)

data_pipeline = DataPipeline(data_source, validator, preprocessor)

factory = MetricCalculatorFactory()
calculators = [
    factory.create_calculator("excess_return"),
    factory.create_calculator("beta"),
    factory.create_calculator("volatility"),
    factory.create_calculator("sharpe_ratio", risk_free_rate=0.03),
    factory.create_calculator("information_ratio"),
]
metric_pipeline = MetricCalculationPipeline(calculators)

standardizer = StandardizerFactory.create_standardizer("zscore")
aggregator = WeightedSumAggregator()
scoring_pipeline = ScoringPipeline(standardizer, aggregator)

metric_columns = ["Excess_Return", "Sharpe_Ratio", "Information_Ratio"]
weights = {"Excess_Return": 0.4, "Sharpe_Ratio": 0.3, "Information_Ratio": 0.3}

orchestrator = PerformanceAnalysisOrchestrator(
    data_pipeline=data_pipeline,
    metric_pipeline=metric_pipeline,
    scoring_pipeline=scoring_pipeline,
)

orchestrator.run_analysis(metric_columns, weights)
