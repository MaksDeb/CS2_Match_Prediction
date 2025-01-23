from zenml import pipeline
from steps import createplots

@pipeline(enable_cache=False)
def createplots_pipeline():
    createplots.create_accuracy_plot()
    createplots.create_precision_plot()
    createplots.create_recall_plot()
    createplots.create_f1_score_plot()
    createplots.create_time_plot()
    createplots.create_extra_trees_plot()
    createplots.create_random_forest_plot()
    createplots.create_models_comparison_histogram()
    createplots.create_nn_time_plot()