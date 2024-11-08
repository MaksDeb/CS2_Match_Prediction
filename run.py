from NeuralNetworkPipeline import datapipeline, dataengineeringpipeline
from NeuralNetworkSteps import dataengineering
from zenml.client import Client
import pandas as pd
from uuid import UUID
from zenml import load_artifact
from zenml import save_artifact


dataloader = datapipeline.datapipeline()

#client = Client()

#latest_run = client.get_pipeline('datapipeline').runs[-1]
#df_artifact = latest_run.steps['load_dataset'].output
#df = load_artifact(df_artifact.name)
#print(df.shape)

#dropcolumns_step = dataengineering.dropcolumns(df=df,column1='team1',column2='team2')
#df = df.drop('team1', axis=1)
#print(df.head())
#dataengineeringpipeline_instance = dataengineeringpipeline.dataengineeringpipeline(
#    drop_step=dataengineering.dropcolumns(artifact_name=df_artifact.name, column1='team1', column2='team2')
#)
dataengineeringpipeline_instance = dataengineeringpipeline.dataengineeringpipeline()
