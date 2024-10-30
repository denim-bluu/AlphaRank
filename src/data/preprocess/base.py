import pandas as pd


from abc import ABC, abstractmethod


class PreprocessingStep(ABC):
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing step to the LazyFrame.

        Args:
            df (pd.DataFrame): Input data.

        Returns:
            Processed LazyFrame
        """
        raise NotImplementedError
