import pandas as pd
import click
from sklearn.model_selection import train_test_split
from module_9_project.config import *


def get_dataset():
    data = pd.read_csv(PATH + "train.csv")
    features = data.drop("Cover_Type", axis=1)
    target = data["Cover_Type"]
    return data


def dataset_split(random_state: int, test_split_ratio: float):
    data = pd.read_csv(PATH + "train.csv")
    click.echo(f"Dataset shape: {data.shape}.")
    features = data.drop("Cover_Type", axis=1)
    target = data["Cover_Type"]
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_split_ratio, random_state=random_state
    )
    return features_train, features_val, target_train, target_val
