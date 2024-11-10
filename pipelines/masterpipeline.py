from zenml import pipeline
from pipelines import dataengineeringpipeline, datapipeline, splittinddatapipeline, trainandpredictpipelineneuralnetwork


@pipeline(enable_cache=False)
def masterpipeline():
    dataloader = datapipeline.datapipeline.with_options(
        config_path='dataloaderconfig.yml')()
    dataengineeringpipeline_instance = dataengineeringpipeline.dataengineeringpipeline.with_options(
        config_path='dataengineeringconfig.yml')()
    splittingdatapipeline_instance = splittinddatapipeline.splittingdatapipeline.with_options(
        config_path='splittingdataconfig.yml')()
    trainandpredictpipeline_instace = trainandpredictpipelineneuralnetwork.trainandpredictpipeline()
