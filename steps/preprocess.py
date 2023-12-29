import os

import dvc.api
import mlflow
import typer
from mlflow.data.pandas_dataset import PandasDataset

from src import files, text_ops
from src.utils.mlflow_run_decorator import mlflow_run


@mlflow_run
def preprocess(filepath: str, dataset: str = "train"):
    df = files.load_dataset(filepath)

    params = dvc.api.params_show()
    import ipdb

    print(params)
    df["keyword"] = df["keyword"].apply(text_ops.clean_keyword)
    df["profile_tags"] = df["text"].apply(lambda x: text_ops.get_text_tags(x, r"@\w+"))
    df["hash_tags"] = df["text"].apply(lambda x: text_ops.get_text_tags(x, r"#\w+"))

    df["link_tags"] = df["text"].apply(
        lambda x: text_ops.get_text_tags(x, r"https://t.co/\w+")
    )
    df["location"] = df["location"].fillna("")
    df.set_index("id", inplace=True)
    # ipdb.set_trace()

    output_filepath = os.path.join("./data/preprocess", f"{dataset}.parquet")
    files.save_dataset(df, output_filepath)

    print("preprocess.run_id", mlflow.active_run().info.run_id)
    mlflow.log_input(
        mlflow.data.from_pandas(df, source=output_filepath), context=dataset
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", help="Path to csv file to be processed")
    parser.add_argument("-d", "--dataset", help="Which dataset")

    args = parser.parse_args()

    preprocess(args.filepath, args.dataset)
