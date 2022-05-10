import click
import os
from joblib import load
from pathlib import Path
import pandas as pd

from module_9_project.data import get_dataset


@click.command()
@click.option(
    "-t",
    "--test-path",
    default="data/test.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
    help="path to the test data",
)
@click.option(
    "-s",
    "--saved-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
    help="path to the saved model",
)
@click.option(
    "-p",
    "--prediction-path",
    default="data",
    type=click.STRING,
    show_default=True,
    help="path where to save the predictions",
)
def predict_model(test_path: Path, saved_model_path: Path, prediction_path: str):
    test_data = get_dataset(test_path)
    saved_model = load(saved_model_path)
    y_predicted = saved_model.predict(test_data)

    submit_df = pd.DataFrame(columns=["Id", "Cover_Type"])
    submit_df["Id"] = test_data.reset_index()["Id"]
    submit_df["Cover_Type"] = y_predicted
    submit_df.to_csv(os.path.join(prediction_path, "submit.csv"), index=False)
