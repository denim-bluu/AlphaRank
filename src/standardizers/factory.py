from typing import Dict, Type

from src.generics.factory import GenericFactory
from .base import MetricStandardizer
from .impl.non_parametric import MinMaxStandardizer


class StandardizerFactory(GenericFactory[MetricStandardizer]):
    """Factory for creating standardizers."""

    _registry: Dict[str, Type[MetricStandardizer]] = {
        "MinMax": MinMaxStandardizer,
    }
