from zenml import pipeline
from NeuralNetworkSteps import dataloader, datainfo


@pipeline(enable_cache=False)
def datapipeline(path: str):
    df = dataloader.load_dataset(path=path)
    datainfo.getdatainfo(df)
    datainfo.getdistributionplot(df)
