from zenml import pipeline
from NeuralNetworkSteps import dataengineering
from zenml.client import Client
from zenml import load_artifact
from uuid import UUID
import pandas as pd


@pipeline
def dataengineeringpipeline():
    client = Client()

    latest_run = client.get_pipeline('datapipeline').runs[-1]
    df_artifact = latest_run.steps['load_dataset'].output

    df = client.get_artifact_version(name_id_or_prefix=df_artifact.name)

    dataengineering.dropcolumns(df=df, column1='team1',column2='team2')
