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
                    X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series, X: pd.DataFrame, y: pd.Series):

    model1_wrapped = KerasClassifier(model=model1)
    model1_wrapped.fit(X_train, y_train)
    #model2.fit(X_test, y_test)
    #model3.fit(X_test, y_test)

    ensemble_model = VotingClassifier(
        estimators=[
            ('neural_network', model1_wrapped),
            ('random_forest', model2),
            ('extra_tree', model3)
        ],
        voting='hard'
    )

    ensemble_model.fit(X_train, y_train)
    y_pred = ensemble_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")

