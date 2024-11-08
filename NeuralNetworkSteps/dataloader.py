import pandas as pd
from zenml import step


@step
def load_dataset():
    path = 'Data/CS2_HLTV_MATCH_DATA2.csv'
    df = pd.read_csv(path, sep=';', parse_dates=False)
    print(df.shape)
    print(df.head())
    return df
