from zenml import step
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


@step(enable_cache=False)
def evaluate_randomoforest_withcv(clf: RandomForestClassifier, cv_scores: np.ndarray, X_train: pd.DataFrame,
                                  y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):

    print(f'Cross-validation scores: {cv_scores}')
    print(f'Mean CV accuracy: {np.mean(cv_scores):.4f}')
    print(f'Standard deviation of CV accuracy: {np.std(cv_scores):.4f}')

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy:  {accuracy}')

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Random Forest Confusion Matrix')
    plt.savefig('DataPlots/randomforest_confusionmatrix')
    plt.show()

