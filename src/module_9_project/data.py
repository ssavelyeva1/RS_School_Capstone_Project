import pandas as pd
from module_9_project.config import *


def get_dataset():
    data = pd.read_csv(PATH + "train.csv")
    features = data.drop("Cover_Type", axis=1)
    target = data["Cover_Type"]
    return data



