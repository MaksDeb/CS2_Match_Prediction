from zenml import step
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import numpy as np


@step(enable_cache=False)
def ensemble_models(model1, model2, model3, X_test, y_test):
    ensemble_model = VotingClassifier(
        estimators=[
            ('model_1', model1),
            ('model_2', model2),
            ('model_3', model3)
        ],
        voting='hard'
    )
    ensemble_model.fit(X_test, y_test)

    y_pred = ensemble_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Dokładność zestawu klasyfikatorów: {accuracy}")
