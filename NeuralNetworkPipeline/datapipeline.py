from zenml import pipeline
from NeuralNetworkSteps import dataloader, datainfo


@pipeline(enable_cache=False)
def datapipeline():
    df = dataloader.load_dataset()
    datainfo.getdatainfo(df)
    datainfo.getdistributionplot(df)