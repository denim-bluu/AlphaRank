from typing import Dict, Type

from .base import MetricCalculator
from .performance_metrics import BetaCalculator, ExcessReturnCalculator
from .ratio_metrics import InformationRatioCalculator, SharpeRatioCalculator
from .risk_metrics import TrackingErrorCalculator, VolatilityCalculator


class MetricCalculatorFactory:
    """Factory for creating metric calculators."""

    _calculators: Dict[str, Type[MetricCalculator]] = {
        "excess_return": ExcessReturnCalculator,
        "beta": BetaCalculator,
        "volatility": VolatilityCalculator,
        "tracking_error": TrackingErrorCalculator,
        "sharpe_ratio": SharpeRatioCalculator,
        "information_ratio": InformationRatioCalculator,
    }

    @classmethod
    def register_calculator(cls, name: str, calculator: Type[MetricCalculator]):
        """
        Register a new calculator type.

        Args:
            name (str): Name of the calculator.
            calculator (Type[MetricCalculator]): Calculator class to register.
        """
        cls._calculators[name] = calculator

    @classmethod
    def create_calculator(cls, calculator_type: str, **kwargs) -> MetricCalculator:
        """
        Create a calculator instance.

        Args:
            calculator_type (str): Type of calculator to create.
            **kwargs: Additional arguments for the calculator constructor.

        Returns:
            MetricCalculator: Instance of the requested calculator.

        Raises:
            ValueError: If the calculator type is unknown.
        """
        calculator_class = cls._calculators.get(calculator_type)
        if calculator_class is None:
            raise ValueError(f"Unknown calculator type: {calculator_type}")
        return calculator_class(**kwargs)

    @classmethod
    def get_calculators(cls) -> Dict[str, Type[MetricCalculator]]:
        return cls._calculators
