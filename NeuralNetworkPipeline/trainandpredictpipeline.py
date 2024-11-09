from zenml import pipeline
from zenml.client import Client
from NeuralNetworkSteps import trainandpredictnn
import pandas as pd


@pipeline(enable_cache=False)
def trainandpredictpipeline():
    client = Client()

    latest_run = client.get_pipeline('splittingdatapipeline').runs[-1]
    print(latest_run.steps)
    df_artifact = latest_run.steps['traintestsplit'].outputs

    X_train = client.get_artifact_version(name_id_or_prefix='splittingdatapipeline::traintestsplit::output_0')
    X_test = client.get_artifact_version(name_id_or_prefix='splittingdatapipeline::traintestsplit::output_1')
    y_train = client.get_artifact_version(name_id_or_prefix='splittingdatapipeline::traintestsplit::output_2')
    y_test = client.get_artifact_version(name_id_or_prefix='splittingdatapipeline::traintestsplit::output_3')

    model = trainandpredictnn.compileneuralnetwork(X_train)
    trainandpredictnn.plotmodel(model)
    trainedmodel = trainandpredictnn.trainmodel(model=model, X_train=X_train, y_train=y_train,
                                                X_test=X_test, y_test=y_test)
    trainandpredictnn.summarizemodel(trainedmodel)
