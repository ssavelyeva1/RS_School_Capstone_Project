from pathlib import Path
from joblib import dump
import numpy as np
import click
import mlflow
import mlflow.sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from .data import dataset_split
from .pipeline import create_pipeline


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
    help='path to the dataset'
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
    help='path to save the model'
)
@click.option(
    "-r",
    "--random-state",
    default=42,
    type=int,
    show_default=True
)
@click.option(
    "-t",
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True
)
@click.option(
    "-mod",
    "--model_name",
    default="random_forest",
    type=click.STRING,
    show_default=True,
    help='model name: random_forest / extra_trees'
)
@click.option(
    "-scale",
    "--scaler_type",
    default="standard",
    type=str,
    show_default=True,
    help='scaler type: standard / min_max'
)
@click.option(
    "-max",
    "--min_samples_split",
    default=2,
    type=int,
    show_default=True,
    help='minimal samples parameter value'
)
@click.option(
    "-crit",
    "--criterion",
    default="gini",
    type=click.STRING,
    show_default=True,
    help='criterion: gini / entropy'
)
@click.option(
    "-max",
    "--max_depth",
    default=100,
    type=int,
    show_default=True,
    help='maximal depth parameter value'
)
def train_model(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    model_name: str,
    scaler_type: str,
    min_samples_split: int,
    criterion: str,
    max_depth: int
):
    x_train, x_test, y_train, y_test = dataset_split(dataset_path, random_state, test_split_ratio)

    experiment_name = "forest_experiment"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(name=experiment_name)

    cv_inner = KFold(n_splits=3, random_state=random_state, shuffle=True)
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=random_state)
    pipe = create_pipeline(model_name, scaler_type, min_samples_split, criterion, max_depth, random_state)

    space = {
        "reg__min_samples_split": [2, 3, 4],
        "reg__criterion": ["gini", "entropy"],
        "reg__max_depth": [100, 75, 125]
    }

    result = GridSearchCV(pipe, space, scoring='accuracy', n_jobs=1, cv=cv_inner, refit=True).fit(x_train, y_train)
    best_model = result.best_estimator_
    best_parameters = result.best_params_

    run_name = model_name + ", " + scaler_type + " scaler"
    with mlflow.start_run(run_name=run_name):
        accuracy = np.max(cross_val_score(best_model, x_train, y_train, scoring='accuracy', cv=cv_outer))
        f1 = np.max(cross_val_score(best_model, x_train, y_train, scoring='f1_micro', cv=cv_outer))
        precision = np.max(cross_val_score(best_model, x_train, y_train, scoring='precision_micro', cv=cv_outer))

        model_parameters = best_parameters
        model_metrics = {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision
        }

        mlflow.log_params(model_parameters)
        mlflow.log_metrics(model_metrics)

        click.echo(f"Accuracy: {accuracy}")
        dump(pipe, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
