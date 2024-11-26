from zenml import step
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from keras._tf_keras.keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import pandas as pd


@step(enable_cache=False)
def ensemble_models(model1: Sequential, model2: RandomForestClassifier, model3: ExtraTreesClassifier,
                    X_test: pd.DataFrame, y_test: pd.Series):

    model1_wrapped = KerasClassifier(model=model1)
    model1_wrapped.fit(X_test, y_test)

    ensemble_model = VotingClassifier(
        estimators=[
            ('neural_network', model1_wrapped),
            ('random_forest', model2),
            ('extra_tree', model3)
        ],
        voting='hard'
    )
    ensemble_model.fit(X_test, y_test)

    y_pred = ensemble_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Dokładność zestawu klasyfikatorów: {accuracy}")

    print(f"y_test: {y_test[:10]}")
    print(f"y_pred: {y_pred[:10]}")

    print("Pierwsze 10 predykcji (y_pred):", y_pred[:10])
    print("Pierwsze 10 wartości rzeczywiste (y_test):", y_test.head(10).tolist())


