from pathlib import Path
import click
from sklearn import ensemble
from sklearn import metrics
from sklearn.metrics import accuracy_score
from .data import dataset_split


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
    "--n_estimators",
    default=100,
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
    n_estimators: int,
    criterion: str,
    max_depth: int
):
    x_train, x_test, y_train, y_test = dataset_split(
        dataset_path,
        random_state,
        test_split_ratio
    )
    rf_clf = ensemble.RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)
    rf_clf.fit(x_train, y_train)
    y_predicted = rf_clf.predict(x_test)
    click.echo(metrics.accuracy_score(y_test, y_predicted))