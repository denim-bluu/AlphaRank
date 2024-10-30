from src.config import DataConfig, ModelConfig
from src.data.preprocess.factory import PreprocessorStepFactory
from src.data.preprocess.preprocessor import DataPreprocessor
from src.data.source import ParquetDataSource
from src.data.validator import DataValidator
from src.pipeline import ModelPipeline
from src.data.pipeline import DataPipeline

data_config = DataConfig()
model_config = ModelConfig()

data_source = ParquetDataSource(data_config.data_file_path)
validator = DataValidator()
preprocessor = DataPreprocessor(
    [PreprocessorStepFactory.create(step) for step in data_config.preprocessor_steps]
)
data_pipeline = DataPipeline(data_source, validator, preprocessor)
data = data_pipeline.run()

model_pipeline = ModelPipeline(model_config)
model_pipeline.run(data)
print(model_pipeline.pm_scores)
