import pandas as pd
from sklearn.metrics import accuracy_score
from zenml import step
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import numpy as np
import time


@step(enable_cache=False)
def trainrf_withcv(X: pd.DataFrame, y: pd.Series, random_state: int,
                   n_splits: int) -> Tuple[RandomForestClassifier, np.ndarray]:
    clf = RandomForestClassifier(random_state=random_state)
    start = time.time()
    k_folds = KFold(n_splits=n_splits)

    cv_scores = cross_val_score(clf, X, y, cv=k_folds)
    end = time.time()
    print(f"Random Forest training took {end - start} seconds")
    print(type(cv_scores))

    return clf, cv_scores
