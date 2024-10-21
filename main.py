from src.data import preprocessor as pp
from src.data.pipeline import DataPipeline
from src.data.source import ParquetDataSource
from src.data.validator import DataValidator
from src.metrics.aggregator.weighted_sum import WeightedSumAggregator
from src.metrics.calculator.factory import MetricCalculatorFactory
from src.metrics.calculator.pipeline import MetricCalculationPipeline
from src.metrics.scoring_pipeline import ScoringPipeline
from src.metrics.standardizer.factory import StandardizerFactory

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

pipeline = DataPipeline(data_source, validator, preprocessor)
processed_data = pipeline.run()

factory = MetricCalculatorFactory()
calculators = [
    factory.create_calculator("excess_return"),
    factory.create_calculator("beta"),
    factory.create_calculator("volatility"),
    factory.create_calculator("sharpe_ratio", risk_free_rate=0.03),
    factory.create_calculator("information_ratio"),
]
pipeline = MetricCalculationPipeline(calculators)
calculated_metrics = pipeline.run(processed_data)

standardizer = StandardizerFactory.create_standardizer("zscore")
aggregator = WeightedSumAggregator()
pipeline = ScoringPipeline(standardizer, aggregator)

metric_columns = ["Excess_Return", "Sharpe_Ratio", "Information_Ratio"]
weights = {"Excess_Return": 0.4, "Sharpe_Ratio": 0.3, "Information_Ratio": 0.3}

result = pipeline.run(calculated_metrics, metric_columns, weights)
final_result = result.collect()
