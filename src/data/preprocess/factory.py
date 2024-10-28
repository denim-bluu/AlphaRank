from src.data.preprocess.base import PreprocessingStep
from src.data.preprocess import steps
from src.generics.factory import GenericFactory


class PreprocessorStepFactory(GenericFactory[PreprocessingStep]):
    _registry: dict[str, type[PreprocessingStep]] = {
        "OptimizeSchemaStep": steps.OptimizeSchemaStep,
        "SortStep": steps.SortStep,
        "RollingMeanReturnStep": steps.RollingMeanReturnStep,
        "RollingMeanBenchmarkReturnStep": steps.RollingMeanBenchmarkReturnStep,
        "RollingStdReturnStep": steps.RollingStdReturnStep,
        "RollingStdBenchmarkReturnStep": steps.RollingStdBenchmarkReturnStep,
        "CumulativeReturnStep": steps.CumulativeReturnStep,
    }
