from zenml import pipeline
from steps import dataengineering
from zenml.client import Client
from zenml import load_artifact
from uuid import UUID
import pandas as pd


@pipeline(enable_cache=False)
def dataengineeringpipeline(column1: str, column2: str,
                            column3: str, column4: str, columnmap: str):
    client = Client()

    latest_run = client.get_pipeline('datapipeline').runs[-1]
    df_artifact = latest_run.steps['load_dataset'].output
    df = client.get_artifact_version(name_id_or_prefix=df_artifact.name)

    df1 = dataengineering.dropcolumns(df=df, column1=column1, column2=column2, column3=column3, column4=column4)
    df2 = dataengineering.decodemapcolumn(df=df1, column=columnmap)
    df3 = dataengineering.changewinratetofloat(df=df2)
    df4 = dataengineering.fillemptyrows(df=df3)
    df5 = dataengineering.droprating1_0_stats(df=df4)
    df_final = dataengineering.calculateavgstat(df=df5)

