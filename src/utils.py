import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import consts as consts


def normalize_data(df_data: pd.DataFrame, type_normalization: str) -> pd.DataFrame:
    if type_normalization == 'standard':
        df_data_normalized = StandardScaler().fit_transform(df_data)
    elif type_normalization == 'minmax':
        df_data_normalized = MinMaxScaler().fit_transform(df_data)
    else:
        df_data_normalized = StandardScaler().fit_transform(df_data)

    return df_data_normalized


def load_dataset():
    # Load data with features extracted
    path_dataset = str(Path.joinpath(consts.PATH_PROJECT_DATA, 'Complete_database_{}.xlsx'.format('fibro')))
    df_data = pd.read_excel(path_dataset, index_col=0)

    # Remove patients with missing data in the acquisition data procedure
    df_data = df_data.drop([12, 18, 23, 24], axis=0)

    X = df_data[df_data['Fibromialgia'] == 1]
    # X = X.drop(['Fibromialgia', 'Pulf'], axis=1)
    X = X.drop(['Fibromialgia', 'Pulf', 'Ortostatismo'], axis=1)
    # X = X.drop(['Fibromialgia', 'Pulf', 'Ortostatismo', 'Sexo'], axis=1)

    print(X.head())
    print(X.shape)

    return X