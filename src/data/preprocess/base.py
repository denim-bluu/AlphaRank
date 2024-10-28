import polars as pl


from abc import ABC, abstractmethod


class PreprocessingStep(ABC):
    @abstractmethod
    def apply(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply preprocessing step to the LazyFrame.

        Args:
            lf: Input LazyFrame to process

        Returns:
            Processed LazyFrame
        """
        raise NotImplementedError
