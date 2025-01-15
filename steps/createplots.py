from zenml import step
import matplotlib.pyplot as plt
import numpy as np

categories = ['Extra Trees', 'Sieć Neuronowa', 'Random Forest', 'Zestaw klasyfikatorów']
accuracy = [72, 74, 71, 73]
precision = [72, 74, 71, 71]
recall = [71, 73, 71, 72]
f1_score = [71, 73, 71, 72]

accuracy = [val / 100 for val in accuracy]
precision = [val / 100 for val in precision]
recall = [val / 100 for val in recall]
f1_score = [val / 100 for val in f1_score]

@step(enable_cache=False)
def create_accuracy_plot():
    plt.figure(figsize=(10, 6))
    plt.bar(categories, accuracy, color='blue', alpha=0.7)
    plt.title('Accuracy (Skuteczność)')
    plt.ylabel('Wartość')
    plt.ylim(0, 1)
    plt.savefig('DataPlots/accuracy.png')
    plt.show()

@step(enable_cache=False)
def create_precision_plot():
    plt.figure(figsize=(10, 6))
    plt.bar(categories, precision, color='green', alpha=0.7)
    plt.title('Precision (Precyzja)')
    plt.ylabel('Wartość')
    plt.ylim(0, 1)
    plt.savefig('DataPlots/precision.png')
    plt.show()

@step(enable_cache=False)
def create_recall_plot():
    plt.figure(figsize=(10, 6))
    plt.bar(categories, recall, color='orange', alpha=0.7)
    plt.title('Recall (Czułość)')
    plt.ylabel('Wartość')
    plt.ylim(0, 1)
    plt.savefig('DataPlots/recall.png')
    plt.show()

@step(enable_cache=False)
def create_f1_score_plot():
    plt.figure(figsize=(10, 6))
    plt.bar(categories, f1_score, color='red', alpha=0.7)
    plt.title('F1-Score (F1-Miara)')
    plt.ylabel('Wartość')
    plt.ylim(0, 1)
    plt.savefig('DataPlots/f1measure.png')
    plt.show()
