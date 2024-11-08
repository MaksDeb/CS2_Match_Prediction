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
    print(f"Artifact Name: {df_artifact.name}")
    print(f"Artifact URI: {df_artifact.uri}")
    print(f"Artifact URI: {df_artifact.id}")
    print(f"Artifact Version: {df_artifact.version}")

    df = client.get_artifact_version(name_id_or_prefix=df_artifact.name)

    print(f"Loaded DataFrame Shape: {df.shape}")

    dataengineering.dropcolumns(df=df, column1='team1',column2='team2')
