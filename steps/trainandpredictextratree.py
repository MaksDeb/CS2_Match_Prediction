import pandas as pd
from zenml import step
from sklearn.ensemble import ExtraTreesClassifier
from typing import Tuple
import time


@step(enable_cache=False)
def train_extra_tree(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int,
                     criterion: str) -> ExtraTreesClassifier:
    extra_tree_forest = ExtraTreesClassifier(n_estimators=n_estimators,
                                             criterion=criterion)
    start = time.time()
    extra_tree_forest.fit(X_train, y_train)
    end = time.time()
    print(f"Extra trees training took {end - start} seconds")
    return extra_tree_forest
