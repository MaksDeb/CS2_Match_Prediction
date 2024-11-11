from zenml import pipeline
from pipelines import dataengineeringpipeline, datapipeline, splittinddatapipeline, trainpipelineneuralnetwork
from pipelines import neuralnetwork_modelevaluationpipeline


@pipeline(enable_cache=False)
def masterpipeline():
    dataloader = datapipeline.datapipeline.with_options(
        config_path='dataloaderconfig.yml')()
    dataengineeringpipeline_instance = dataengineeringpipeline.dataengineeringpipeline.with_options(
        config_path='dataengineeringconfig.yml')()
    splittingdatapipeline_instance = splittinddatapipeline.splittingdatapipeline.with_options(
        config_path='splittingdataconfig.yml')()
    trainandpredictpipeline_instace = trainandpredictpipelineneuralnetwork.train_neuralnetwork_pipeline()
    modelevaluationpipeline_instance = neuralnetwork_modelevaluationpipeline.neuralnetwork_modelevaluationpipeline()

