from .base import MetricStandardizer
from .non_parametric import MinMaxStandardizer
from .parametric import ZScoreStandardizer


class StandardizerFactory:
    """Factory for creating standardizers."""

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
        if standardizer_type == "zscore":
            return ZScoreStandardizer()
        elif standardizer_type == "minmax":
            return MinMaxStandardizer()
        else:
            raise ValueError(f"Unknown standardizer type: {standardizer_type}")
