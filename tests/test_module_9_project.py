from pathlib import Path
import click
from click.testing import CliRunner
from joblib import load
import pytest

import pandas as pd
from sklearn.pipeline import Pipeline

from module_9_project.data import get_dataset, dataset_split, dataset_train_test_split
from module_9_project.train import train_model
from module_9_project.pipeline import create_pipeline


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_pipeline_return() -> None:
    """Tests the returned object of the pipeline function"""
    pipeline = create_pipeline("random_forest", "standard", 2, "gini", 100, 42)
    assert isinstance(pipeline, Pipeline)


def test_get_dataset_return() -> None:
    """Tests the returned object of the get_dataset function"""
    data = get_dataset("tests/test_data.csv")
    assert isinstance(data, pd.DataFrame)


def test_dataset_split_return() -> None:
    """Tests the returned object of the dataset_split function"""
    features, target = dataset_split("tests/test_data.csv")
    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, pd.Series)


def test_error_for_invalid_test_split_ratio() -> None:
    """Tests fails when test split ratio is greater than 1 in dataset_train_test_split function"""
    file_path = "tests/test_data.csv"
    random_state = 42
    test_split_ratio = 24
    with pytest.raises(ValueError):
        dataset_train_test_split(file_path, random_state, test_split_ratio)


def test_dataset_train_test_split_return() -> None:
    """Tests the returned object of the dataset_train_test_split function"""
    features_train, features_val, target_train, target_val = dataset_train_test_split(
        file_path="tests/test_data.csv",
        random_state=42,
        test_split_ratio=0.2
    )
    assert isinstance(features_train, pd.DataFrame)
    assert isinstance(features_val, pd.DataFrame)
    assert isinstance(target_train, pd.Series)
    assert isinstance(target_val, pd.Series)


def test_error_for_invalid_train_model_path(
    runner: CliRunner
) -> None:
    """Test fails when path for function train_model is invalid."""
    result = runner.invoke(
        train_model,
        [
            "--dataset-path",
            "incorrect/path/train.csv"
        ],
    )
    assert result.exit_code == 2
    assert "Error: Invalid value for '-d' / '--dataset-path': File 'incorrect/path/train.csv' " \
           "does not exist." in result.output


def test_saving_path_for_train_model(
        runner: CliRunner
) -> None:
    """Test checks where model is saved for train_model."""
    result = runner.invoke(
        train_model,
        [
            "--dataset-path",
            "tests/test_data.csv",
            "--save-model-path",
            "tests/model.joblib"
        ],
    )
    saved_model = Path("tests/model.joblib")
    assert saved_model.exists()
    loaded_model = load("tests/model.joblib")
    assert isinstance(loaded_model, Pipeline)


