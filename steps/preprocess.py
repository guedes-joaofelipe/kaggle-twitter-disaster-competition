import os

import dvc.api
import mlflow
import pandas as pd
import typer
from sklearn.model_selection import train_test_split

from src import files, text_ops
from src.decorators import mlflow_run


@mlflow_run
def preprocess(filepath: str, dataset: str):
    df = files.load_dataset(filepath)

    params = dvc.api.params_show()

    df["keyword"] = df["keyword"].apply(text_ops.clean_keyword)
    df["profile_tags"] = df["text"].apply(lambda x: text_ops.get_text_tags(x, r"@\w+"))
    df["hash_tags"] = df["text"].apply(lambda x: text_ops.get_text_tags(x, r"#\w+"))

    df["link_tags"] = df["text"].apply(
        lambda x: text_ops.get_text_tags(x, r"https://t.co/\w+")
    )
    df["location"] = df["location"].fillna("")
    df.set_index("id", inplace=True)

    if dataset == "train":
        df_train, df_valid = train_test_split(
            df,
            test_size=params["evaluate"]["valid_size"],
            random_state=params["preprocess"]["random_state"],
            shuffle=params["preprocess"]["split"]["shuffle"],
        )
        save_dataset(df_train, "train")
        save_dataset(df_valid, "valid")

    else:
        save_dataset(df, dataset)


def save_dataset(df: pd.DataFrame, dataset: str):
    output_filepath = os.path.join("./data/preprocess", f"{dataset}.parquet")
    files.save_dataset(df, output_filepath)

    mlflow.log_input(
        mlflow.data.from_pandas(df, source=output_filepath), context=dataset
    )


if __name__ == "__main__":
    typer.run(preprocess)
