import click
import os
from joblib import load
import pandas as pd

from module_9_project.data import get_dataset


@click.command()
def predict_model():
    test_data = get_dataset("data/test.csv")
    saved_model = load("data/model.joblib")
    y_predicted = saved_model.predict(test_data)

    submit_df = pd.DataFrame(columns=["Id", "Cover_Type"])
    submit_df["Id"] = test_data.reset_index()["Id"]
    submit_df["Cover_Type"] = y_predicted
    submit_df.to_csv(os.path.join("data", "submit.csv"), index=False)


