from zenml import step
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from keras._tf_keras.keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@step(enable_cache=False)
def ensemble_models(model1: Sequential, model2: RandomForestClassifier, model3: ExtraTreesClassifier,
                    X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series, X: pd.DataFrame, y: pd.Series):

    model1_wrapped = KerasClassifier(model=model1)
    model1_wrapped.fit(X_train, y_train)

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

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Ensemble models Confusion Matrix')
    plt.savefig('DataPlots/ensemble_models_confusionmatrix')
    plt.show()

