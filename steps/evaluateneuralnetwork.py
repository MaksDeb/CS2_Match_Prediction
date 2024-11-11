from zenml import step
from keras._tf_keras.keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


@step(enable_cache=False)
def evaluate_neuralnetwork(model: Sequential, X_test: pd.DataFrame, y_test: pd.Series):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')

    y_pred_probs = model.predict(X_test)
    y_pred = np.round(y_pred_probs).astype(int).flatten()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Neural Network Confusion Matrix')
    plt.savefig('DataPlots/neuralnetwork_confusionmatrix')
    plt.show()
