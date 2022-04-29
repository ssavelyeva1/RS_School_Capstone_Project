import pandas as pd
import click
from pathlib import Path
from sklearn.model_selection import train_test_split


def get_dataset(file_path: Path):
    data = pd.read_csv(file_path)
    features = data.drop("Cover_Type", axis=1)
    target = data["Cover_Type"]
    return data


def dataset_split(file_path: Path, random_state: int, test_split_ratio: float):
    data = pd.read_csv(file_path)
    click.echo(f"Dataset shape: {data.shape}.")
    features = data.drop("Cover_Type", axis=1)
    target = data["Cover_Type"]
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_split_ratio, random_state=random_state
    )
    return features_train, features_val, target_train, target_val
