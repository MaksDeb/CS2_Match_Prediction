import pandas as pd
from zenml import step


@step
def load_dataset():
    path = 'Data/CS2_HLTV_MATCH_DATA2.csv'
    df = pd.read_csv(path, sep=';')
    print(df.shape)
    df.head(10)
    return df

