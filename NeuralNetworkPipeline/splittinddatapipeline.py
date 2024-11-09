from zenml import pipeline
from zenml.client import Client
from NeuralNetworkSteps import splittingdata
import pandas as pd


@pipeline(enable_cache=False)
def splittingdatapipeline(train_sample: int, test_sample: int, random_state: int):
    client = Client()

    latest_run = client.get_pipeline('dataengineeringpipeline').runs[-1]
    df_artifact = latest_run.steps['calculateavgstat'].outputs

    dataframe_artifact = None
    for artifact in df_artifact.values():
        if artifact.data_type.attribute == "DataFrame":
            dataframe_artifact = artifact
            break
    print(dataframe_artifact.name)

    df = client.get_artifact_version(name_id_or_prefix=dataframe_artifact.name)

    X_train, X_test, y_train, y_test = splittingdata.traintestsplit(df=df, train_sample=train_sample, test_sample=test_sample,
                                                                    random_state=random_state)
