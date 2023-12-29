import os
import shutil

import pandas as pd


def load_dataset(filepath: str) -> pd.DataFrame:
    extension = filepath.split(".")[-1]

    if extension == "csv":
        df = pd.read_csv(filepath)
    elif extension == "parquet":
        df = pd.read_parquet(filepath)
    else:
        raise NotImplementedError

    return df


def save_dataset(df: pd.DataFrame, filepath: str):
    folder = os.path.join(*filepath.split("/")[:-1])
    if not os.path.exists(folder):
        print("Creating folder", folder)
        os.mkdir(folder)
    return df.to_parquet(filepath)


def remove_dir(dir: str):
    if os.path.exists(dir):
        print("Removing dir", dir)
        shutil.rmtree(dir)
