import pandas as pd
from zenml import step
from sklearn.model_selection import train_test_split
from typing import Tuple


@step
def splitdata(df: pd.DataFrame, columntodrop: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columntodrop, axis=1)
    y = df[columntodrop]
    return X, y


@step
def traintestsplit(X: pd.DataFrame, y: pd.Series, train_sample: int, test_sample: int, random_state: int) -> Tuple[pd.DataFrame,
                                                                                                      pd.DataFrame,
                                                                                                      pd.Series, pd.Series]:

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_sample, test_size=test_sample,
                                                        random_state=random_state, shuffle=True)

    return X_train, X_test, y_train, y_test
