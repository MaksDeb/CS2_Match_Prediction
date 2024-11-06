from zenml import step
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings


@step
def getdatainfo(df):
    df.info()
    class_freq = (df.groupby('team1_win').size())
    print(type(class_freq))
    print(class_freq)
    return None


@step
def getdistributionplot(df):
    warnings.filterwarnings('ignore')
    plt.hist(df['team1_win'])
    plt.xlabel('Team1 result')
    plt.ylabel('Frequency')
    plt.title('Distribution of team1_win column')
    plt.savefig('DataPlots/distribution.png')
    plt.show()
    return None
