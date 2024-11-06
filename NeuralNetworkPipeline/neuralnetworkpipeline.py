from zenml import pipeline
from NeuralNetworkSteps import dataloader, datainfo


@pipeline
def datapipeline():
    df = dataloader.load_dataset()
    #datainfo.getdatainfo(df)
    datainfo.getdistributionplot(df)


