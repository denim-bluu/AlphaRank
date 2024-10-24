from typing import Dict, Type
from .base import MetricStandardizer
from .non_parametric import MinMaxStandardizer
from .parametric import ZScoreStandardizer


class StandardizerFactory:
    """Factory for creating standardizers."""

    _standardizers: Dict[str, Type[MetricStandardizer]] = {
        "zscore": ZScoreStandardizer,
        "minmax": MinMaxStandardizer,
    }

    @staticmethod
    def create_standardizer(standardizer_type: str) -> MetricStandardizer:
        """
        Create a standardizer instance.

        Args:
            standardizer_type (str): Type of standardizer to create ('zscore' or 'minmax').

        Returns:
            MetricStandardizer: Instance of the requested standardizer.

        Raises:
            ValueError: If the standardizer type is unknown.
        """
        standardizer_class = StandardizerFactory._standardizers.get(standardizer_type)
        if standardizer_class is None:
            raise ValueError(f"Unknown standardizer type: {standardizer_type}")
        return standardizer_class()

    @classmethod
    def available_standardizers(cls) -> list[str]:
        """Return a list of available standardizers."""
        return list(cls._standardizers.keys())

    @classmethod
    def register_standardizer(cls, name: str, standardizer: Type[MetricStandardizer]):
        """
        Register a new standardizer type.

        Args:
            name (str): Name of the standardizer.
            standardizer (Type[MetricStandardizer]): Standardizer class to register.
        """
        cls._standardizers[name] = standardizer
