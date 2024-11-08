from zenml import step
import pandas as pd


@step
def dropcolumns(df: pd.DataFrame, column1: str, column2: str) -> pd.DataFrame:
    df = df.drop([column1], axis=1)
    df = df.drop([column2], axis=1)
    return df
