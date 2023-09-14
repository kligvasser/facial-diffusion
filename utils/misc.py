import pickle
import json
import pandas as pd
import os


def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def delete(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def save_dict(d, dict_path):
    with open(dict_path, 'wb') as f:
        pickle.dump(d, f)


def load_dict(dict_path):
    with open(dict_path, 'rb') as f:
        loaded_dict = pickle.load(f)

    return loaded_dict


def save_list_as_json(lst, json_path):
    with open(json_path, 'w') as f:
        json.dump(lst, f)


def load_json(json_path):
    with open(json_path, 'rb') as f:
        loaded_dict = json.load(f)

    return loaded_dict


def save_df(df, df_path):
    ext = df_path.split('.')[-1]

    if ext == 'csv':
        df.to_csv(df_path, index=False)
    elif ext == 'xlsx':
        df.to_excel(df_path, index=False)
    elif ext == 'json':
        df.to_json(df_path, orient='records')
    elif ext == 'parquet':
        df.to_parquet(df_path, index=False)
    elif ext == 'pkl':
        df.to_pickle(df_path)
    else:
        raise ValueError(
            "Unsupported file format. Please choose from 'csv', 'xlsx', 'json', 'parquet', or 'pkl'."
        )


def load_df(df_path):
    ext = df_path.split('.')[-1]

    if ext == 'csv':
        df = pd.read_csv(df_path)
    elif ext == 'xlsx':
        df = pd.read_excel(df_path)
    elif ext == 'json':
        df = pd.read_json(df_path)
    elif ext == 'parquet':
        df = pd.read_parquet(df_path)
    elif ext == 'pkl':
        df = pd.read_pickle(df_path)
    else:
        raise ValueError(
            "Unsupported file format. Please choose from 'csv', 'xlsx', 'json', 'parquet', or 'pickle'."
        )

    return df
