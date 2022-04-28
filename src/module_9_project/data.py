import pandas as pd

PATH = "data/"


def get_dataset():
    data = pd.read_csv(PATH + "train.csv")
    features = data.drop("Cover_Type", axis=1)
    target = data["Cover_Type"]
    return data



