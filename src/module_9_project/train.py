from pathlib import Path
from joblib import dump
import click
import mlflow
import mlflow.sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score

from .data import dataset_split
from .pipeline import create_pipeline


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--min_samples_split",
    default=2,
    type=int,
    show_default=True,
)
@click.option(
    "--criterion",
    default="gini",
    type=click.STRING,
    show_default=True,
)
@click.option(
    "--max_depth",
    default=100,
    type=int,
    show_default=True,
)
def train_model(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    min_samples_split: int,
    criterion: str,
    max_depth: int
):
    x_train, x_test, y_train, y_test = dataset_split(dataset_path, random_state, test_split_ratio)

    with mlflow.start_run():
        pipe = create_pipeline(min_samples_split, criterion, max_depth, random_state)
        pipe.fit(x_train, y_train)
        y_predicted = pipe.predict(x_test)

        accuracy = metrics.accuracy_score(y_test, y_predicted)
        f1_score = metrics.f1_score(y_test, y_predicted)
        precision = metrics.precision_score(y_test, y_predicted)

        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("criterion", criterion)
        mlflow.log_param("max_depth", max_depth)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1_score)
        mlflow.log_metric("precision", precision)

        click.echo(f"Accuracy: {accuracy}")
        dump(pipe, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
