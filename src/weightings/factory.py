from typing import Dict, Type
from src.generics.factory import GenericFactory

from .base import WeightingMethod
from .impl.entropy import EntropyWeighting
from .impl.equal_weights import EqualWeighting


class WeightingMethodFactory(GenericFactory[WeightingMethod]):
    _registry: Dict[str, Type[WeightingMethod]] = {
        EntropyWeighting.__name__: EntropyWeighting,
        EqualWeighting.__name__: EqualWeighting,
    }
