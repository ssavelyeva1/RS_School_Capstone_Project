from pathlib import Path
import click
import mlflow
from sklearn import metrics
from sklearn.metrics import accuracy_score

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
    random_state: int,
    test_split_ratio: float,
    min_samples_split: int,
    criterion: str,
    max_depth: int
):
    x_train, x_test, y_train, y_test = dataset_split(dataset_path, random_state, test_split_ratio)

    pipe = create_pipeline(min_samples_split, criterion, max_depth, random_state)
    pipe.fit(x_train, y_train)
    y_predicted = pipe.predict(x_test)

    click.echo(metrics.accuracy_score(y_test, y_predicted))
