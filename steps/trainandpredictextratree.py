import pandas as pd
from zenml import step
from sklearn.ensemble import ExtraTreesClassifier
from typing import Tuple


@step(enable_cache=False)
def train_extra_tree(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int,
                     criterion: str) -> ExtraTreesClassifier:
    extra_tree_forest = ExtraTreesClassifier(n_estimators=n_estimators,
                                             criterion=criterion)
    extra_tree_forest.fit(X_train, y_train)
    return extra_tree_forest
