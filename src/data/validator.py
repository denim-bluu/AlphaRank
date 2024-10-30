import pandas as pd
from pydantic import BaseModel, field_validator


class PerformanceRecord(BaseModel):
    PM_ID: str
    Strategy_ID: str
    Benchmark_ID: str
    Date: str
    Return: float
    Benchmark_Return: float
    Excess_Return: float

    @field_validator("PM_ID")
    def validate_pm_id(cls, v):
        if not v.startswith("PM_"):
            raise ValueError('PM_ID must start with "PM_"')
        return v

    @field_validator("Strategy_ID")
    def validate_strategy_id(cls, v):
        if not v.startswith("S_"):
            raise ValueError('Strategy_ID must start with "S_"')
        return v

    @field_validator("Return", "Benchmark_Return", "Excess_Return")
    def validate_returns(cls, v):
        if v < -1:
            raise ValueError("Return values cannot be less than -100%")
        return v


class DataValidator:
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        # Pandas version
        return df[
            df["PM_ID"].str.contains(r"^PM_")
            & df["Strategy_ID"].str.contains(r"^S_")
            & (df["Return"] >= -1)
            & (df["Benchmark_Return"] >= -1)
            & (df["Excess_Return"] >= -1)
        ]
