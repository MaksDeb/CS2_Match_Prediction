from zenml import step
import pandas as pd
from collections import defaultdict


@step(enable_cache=False)
def dropcolumns(df: pd.DataFrame, column1='team1', column2='team2',
                column3='Unnamed: 152', column4='matchID') -> pd.DataFrame:
    df = df.drop([column1, column2, column3, column4], axis=1)
    print(f"Artifact from dropcolumns step: {df}")
    return df


@step(enable_cache=False)
def decodemapcolumn(df: pd.DataFrame, column='map') -> pd.DataFrame:
    df = df.drop([column], axis=1)
    print(f"Artifact from dropcolumns step: {df}")
    return df


@step(enable_cache=False)
def changewinratetofloat(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if "%" in column:
            df[column] = df[column].str.rstrip('%').astype('float') / 100

    df['T1_mapwinrate'] = df['T1_mapwinrate'].str.rstrip('%').astype('float') / 100.0
    df['T2_mapwinrate'] = df['T2_mapwinrate'].str.rstrip('%').astype('float') / 100.0
    print(f"Artifact from changwinrates step: {df}")
    return df


@step(enable_cache=False)
def fillemptyrows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace(['∞', 'inf', '-inf'], 0)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)
    print(f"Artifact from fillemptyrows step: {df}")
    return df


@step(enable_cache=False)
def droprating1_0_stats(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if "Rating 1.0" in column:
            df = df.drop(column, axis=1)
    print(f"Artifact from droprating10 step: {df}")
    return df


@step(enable_cache=False)
def calculateavgstat(df: pd.DataFrame) -> pd.DataFrame:
    player_stats_columns = [col for col in df.columns if 'player' in col.lower()]

    team1_stats = defaultdict(list)
    team2_stats = defaultdict(list)

    for col in player_stats_columns:
        if 'T1_' in col:
            stat_type = col.replace('T1_', '').split('_')[1]
            team1_stats[stat_type].append(col)
        elif 'T2_' in col:
            stat_type = col.replace('T2_', '').split('_')[1]
            team2_stats[stat_type].append(col)

    for stat_type, columns in team1_stats.items():
        df[f'T1_avg_{stat_type}'] = df[columns].mean(axis=1, skipna=True)
        df[f'T1_sum_{stat_type}'] = df[columns].sum(axis=1, skipna=True)

    for stat_type, columns in team2_stats.items():
        df[f'T2_avg_{stat_type}'] = df[columns].mean(axis=1, skipna=True)
        df[f'T2_sum_{stat_type}'] = df[columns].sum(axis=1, skipna=True)

    df = df.drop(columns=player_stats_columns)
    print(df.head())
    print(f"Artifact from calculateavgstat step: {df}")
    return df
