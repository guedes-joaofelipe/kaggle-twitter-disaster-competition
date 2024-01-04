import os

import dvc.api
import mlflow
import pandas as pd
import spacy
import typer
from sklearn.model_selection import train_test_split

from src import files, text_ops
from src.decorators import mlflow_run

nlp = spacy.load("en_core_web_sm")


@mlflow_run
def preprocess(filepath: str, dataset: str):
    df = files.load_dataset(filepath)

    params = dvc.api.params_show()

    import ipdb

    ipdb.set_trace()
    df = get_features(df, params)

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


def get_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df["keyword"] = df["keyword"].apply(text_ops.clean_keyword)
    df["profile_tags"] = df["text"].apply(lambda x: text_ops.get_text_tags(x, r"@\w+"))
    df["hash_tags"] = df["text"].apply(lambda x: text_ops.get_text_tags(x, r"#\w+"))

    df["link_tags"] = df["text"].apply(
        lambda x: text_ops.get_text_tags(x, r"https://t.co/\w+")
    )

    for tag_feature in ["profile_tags", "hash_tags", "link_tags"]:
        df[f"n_{tag_feature}"] = df[tag_feature].apply(len)

    character_count = {"exclamation_count": "!", "question_count": "?"}

    for label, character in character_count.items():
        df[label] = df["text"].apply(lambda x: text_ops.count_character(x, character))

    df["location"] = df["location"].fillna("")
    df["location_ner"] = df["location"].apply(
        lambda x: text_ops.get_location_labels(x, nlp)
    )
    df["with_location"] = df["location_ner"].apply(lambda x: int(len(x.keys()) > 0))

    df["text_clean"] = df["text"].apply(text_ops.clean_text)

    return df


def save_dataset(df: pd.DataFrame, dataset: str):
    output_filepath = os.path.join("./data/preprocess", f"{dataset}.parquet")
    files.save_dataset(df, output_filepath)

    mlflow.log_input(
        mlflow.data.from_pandas(df, source=output_filepath), context=dataset
    )


if __name__ == "__main__":
    typer.run(preprocess)
