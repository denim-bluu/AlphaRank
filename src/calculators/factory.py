from typing import Dict, Type

from src.generics.factory import GenericFactory

from .base import MetricCalculator
from .return_metrics import Beta, ExcessReturn
from .risk_adjusted_return_metrics import (
    InformationRatio,
    OmegaRatio,
    SharpeRatio,
)
from .risk_metrics import TrackingError, Volatility


class MetricCalculatorFactory(GenericFactory[MetricCalculator]):
    """Factory for creating metric calculators."""

    _registry: Dict[str, Type[MetricCalculator]] = {
        ExcessReturn.__name__: ExcessReturn,
        Beta.__name__: Beta,
        InformationRatio.__name__: InformationRatio,
        OmegaRatio.__name__: OmegaRatio,
        SharpeRatio.__name__: SharpeRatio,
        TrackingError.__name__: TrackingError,
        Volatility.__name__: Volatility,
    }
