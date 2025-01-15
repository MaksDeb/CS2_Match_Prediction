from zenml import pipeline
from steps import createplots

@pipeline(enable_cache=False)
def createplots_pipeline():
    createplots.create_accuracy_plot()
    createplots.create_precision_plot()
    createplots.create_recall_plot()
    createplots.create_f1_score_plot()