from zenml import pipeline
from zenml.client import Client
from NeuralNetworkSteps import splittingdata


@pipeline(enable_cache=False)
def splittingdatapipeline():
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

    X_train, X_test, y_train, y_test = splittingdata.traintestsplit(df=df)
