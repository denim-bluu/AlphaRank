from typing import Dict, Type

from src.generics.factory import GenericFactory

from .base import MetricCalculator
from .impl import return_metrics as returns
from .impl import risk_adjusted_return_metrics as risk_adjusted
from .impl import risk_metrics as risk


class MetricCalculatorFactory(GenericFactory[MetricCalculator]):
    """Factory for creating metric calculators."""

    _registry: Dict[str, Type[MetricCalculator]] = {
        returns.ExcessReturn.__name__: returns.ExcessReturn,
        returns.Beta.__name__: returns.Beta,
        returns.JensensAlpha.__name__: returns.JensensAlpha,
        risk_adjusted.InformationRatio.__name__: risk_adjusted.InformationRatio,
        risk_adjusted.OmegaRatio.__name__: risk_adjusted.OmegaRatio,
        risk_adjusted.SharpeRatio.__name__: risk_adjusted.SharpeRatio,
        risk.TrackingError.__name__: risk.TrackingError,
        risk.Volatility.__name__: risk.Volatility,
    }
