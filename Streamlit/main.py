import streamlit as st
import pandas as pd
import pickle
import zenml
from tensorflow.python.keras import Sequential
from zenml.client import Client
from zenml.enums import ModelStages


def load_model_from_zenml(model_id: str, stage: ModelStages):
    client = Client()

    model = client.get_model(model_id)
    # model_version = client.get_model_version(
    #     model_name_or_id=model_id,
    #     model_version_name_or_number_or_id=stage
    # )
    #
    # artifact = model_version.model_artifact
    # model_path = artifact.uri
    #
    # print(f"Model path: {model_path}")
    #
    # with open(model_path, "rb") as f:
    #     model = pickle.load(f)

    return model

def load_data_from_zenml(artifact_name: str) -> pd.DataFrame:
    client = Client()
    df  = client.get_artifact_version(name_id_or_prefix=artifact_name)
    print(type(df))
    return df

file_path = 'C://Users//maksd//OneDrive//Pulpit//inzynierka//CS2_Match_Prediction//Data//CS2_HLTV_MATCH_DATA2.csv'
df = pd.read_csv(file_path, delimiter=';')
#df = load_data_from_zenml('dataengineeringpipeline::calculateavgstat::output')



def get_team_stats(team1, team2):
    team1_stats = df[(df['team1'] == team1) | (df['team2'] == team1)]
    team2_stats = df[(df['team1'] == team2) | (df['team2'] == team2)]

    combined_stats = pd.concat([team1_stats, team2_stats], ignore_index=True)

    return combined_stats


st.title("Match Prediction for CS2 Teams")

# Select teams from dropdowns
team1 = st.selectbox("Select Team 1", df['team1'].unique())
team2 = st.selectbox("Select Team 2", df['team2'].unique())

artifact_name = "nn_72%_test_set"
try:
    model = load_model_from_zenml(
        model_id="5e5eddc0-241e-417f-bf6d-e491831c7b30",
        stage=ModelStages.STAGING
    )
    st.success("Model successfully loaded!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

model = Sequential
print(type(model))

if st.button('Predict Match Outcome'):
    stats = get_team_stats(team1, team2)

    print(stats.shape)
    prediction = model.predict(stats)

    if prediction == 1:
        st.write(f"Prediction: {team1} will win! Probability: {prediction}")
    else:
        st.write(f"Prediction: {team2} will win! Probability: {prediction}")
