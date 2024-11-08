from zenml import step
import pandas as pd
from zenml import load_artifact
from zenml.client import Client


@step
def dropcolumns(df: pd.DataFrame, column1: str, column2: str) -> pd.DataFrame:
    #client = Client()

    #df = load_artifact(artifact_name)

    df = df.drop([column1], axis=1)
    df = df.drop([column2], axis=1)
    return df
