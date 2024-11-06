from zenml import pipeline
from NeuralNetworkSteps import dataloader

@pipeline
def loading_data():
    df = dataloader.load_dataset()



