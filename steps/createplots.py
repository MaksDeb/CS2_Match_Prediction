from zenml import step
import matplotlib.pyplot as plt
import numpy as np

categories = ['Extra Trees', 'Sieć Neuronowa', 'Random Forest', 'Zestaw klasyfikatorów']
time_categories = ['Extra Trees', 'Sieć Neuronowa', 'Random Forest']
accuracy = [72, 74, 71, 73]
precision = [72, 74, 71, 71]
recall = [71, 73, 71, 72]
f1_score = [71, 73, 71, 72]
time = [0.1, 17.3, 1.5]


extra_trees_categories = ['criterion - entropy,\nn_estimators - 70', 'criterion - gini,\nn_estimators - 20', 'criterion - gini,\nn-estimators - 70']
extra_trees_accuracy = [71, 68, 72]
extra_trees_precision = [71, 69, 72]
extra_trees_recall = [70, 69, 71]
extra_trees_f1_score = [71, 69, 71]

random_forest_categoreis = ['n_splits - 5, criterion-gini', 'n_splits - 5, criterion-entropy', 'n_splits - 10, criterion-entropy']
random_forest_mean_cv = [71,71,71]
random_forest_accuracy = [71, 69, 69]
random_forest_precision = [71, 71, 71]
random_forest_recall = [71, 70, 70]
random_forest_f1_score = [71, 69, 71]

neural_network_categories = ['50', '80', '130','150']
neural_network_time_per_epoch = [8.08, 11.59, 17.3, 22.24]

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

@step(enable_cache=False)
def create_time_plot():
    plt.figure(figsize=(10, 6))
    plt.bar(time_categories, time, color='yellow', alpha=0.7)
    plt.title('Czas treningu modelu')
    plt.ylabel('Sekundy')
    plt.ylim(0, 20)
    plt.savefig('DataPlots/time.png')
    plt.show()


@step(enable_cache=False)
def create_extra_trees_plot():
    x = np.arange(len(extra_trees_categories))
    bar_width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x, extra_trees_accuracy, bar_width, label='Skuteczność')
    ax.bar(x + bar_width, extra_trees_precision, bar_width, label='Precyzja')
    ax.bar(x + 2 * bar_width, extra_trees_recall, bar_width, label='Czułość')
    ax.bar(x + 3 * bar_width, extra_trees_f1_score, bar_width, label='F1-miara')

    ax.set_xlabel('Hiper parametry', fontsize=12)
    ax.set_ylabel('Wartość', fontsize=12)
    ax.set_title('Porównanie wyników dla różnych hiper parametrów dla Extra Trees', fontsize=14)
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(extra_trees_categories, ha='center')
    ax.legend(title="Metryki")

    plt.savefig('DataPlots/extra_trees_comparison.png')
    plt.show()

@step(enable_cache=False)
def create_random_forest_plot():
    x = np.arange(len(random_forest_categoreis))
    bar_width = 0.1

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x, extra_trees_accuracy, bar_width, label='Skuteczność')
    ax.bar(x + bar_width, random_forest_accuracy, bar_width, label='Precyzja')
    ax.bar(x + 2 * bar_width, random_forest_recall, bar_width, label='Czułość')
    ax.bar(x + 3 * bar_width, extra_trees_f1_score, bar_width, label='F1-miara')
    ax.bar(x + 4 * bar_width, random_forest_mean_cv, bar_width, label='Średnia skuteczność podczas walidacji krzyżowej')

    ax.set_xlabel('Hiper parametry', fontsize=12)
    ax.set_ylabel('Wartość', fontsize=12)
    ax.set_title('Porównanie wyników dla różnych hiper parametrów dla Random Forest', fontsize=14)
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(random_forest_categoreis, ha='center')
    ax.legend(title="Metryki")

    plt.savefig('DataPlots/random_forest_comparison.png')
    plt.show()

@step(enable_cache=False)
def create_models_comparison_histogram():
    x = np.arange(len(categories))
    bar_width = 0.1

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(x, accuracy, bar_width, label='Skuteczność (Accuracy)', alpha=0.7)
    ax.bar(x + bar_width, precision, bar_width, label='Precyzja (Precision)', alpha=0.7)
    ax.bar(x + 2 * bar_width, recall, bar_width, label='Czułość (Recall)', alpha=0.7)
    ax.bar(x + 3 * bar_width, f1_score, bar_width, label='F1-Miara (F1-Score)', alpha=0.7)

    ax.set_xlabel('Modele', fontsize=12)
    ax.set_ylabel('Wartość', fontsize=12)
    ax.set_title('Porównanie wyników modeli', fontsize=14)
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(categories, fontsize=10)

    ax.legend(title="Metryki", fontsize=10)

    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('DataPlots/models_comparison_histogram.png')
    plt.show()

@step(enable_cache=False)
def create_nn_time_plot():
    plt.figure(figsize=(10, 6))
    plt.bar(neural_network_categories, neural_network_time_per_epoch, color='pink', alpha=0.7)
    plt.title('Czas treningu modelu sieci neuronowej w zależności od liczby epok')
    plt.ylabel('Sekundy')
    plt.ylim(0, 25)
    plt.savefig('DataPlots/nn_time.png')
    plt.show()