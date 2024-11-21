from zenml import step
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


@step(enable_cache=False)
def evaluate_extra_tree(clf: ExtraTreesClassifier, X_test: pd.DataFrame, y_test: pd.Series):
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
    plt.title('Extra Tree Confusion Matrix')
    plt.savefig('DataPlots/extratree_confusionmatrix')
    plt.show()
