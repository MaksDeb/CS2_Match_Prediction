from pipelines import datapipeline, dataengineeringpipeline, splittinddatapipeline, trainandpredictpipelineneuralnetwork
from pipelines import masterpipeline, modelevaluationpipeline

dataloader = datapipeline.datapipeline.with_options(
    config_path='dataloaderconfig.yml')()
dataengineeringpipeline_instance = dataengineeringpipeline.dataengineeringpipeline.with_options(
    config_path='dataengineeringconfig.yml')()
splittingdatapipeline_instance = splittinddatapipeline.splittingdatapipeline.with_options(
    config_path='splittingdataconfig.yml')()
trainandpredictpipeline_instace = trainandpredictpipelineneuralnetwork.trainandpredictpipeline()
modelevaluationpipeline_instance = modelevaluationpipeline.modelevaluationpipeline()
