import pandas as pd
import pytest

from module_9_project.data import get_dataset, dataset_split, dataset_train_test_split


def test_get_dataset_return() -> None:
    """Tests the returned object of the get_dataset function"""
    data = get_dataset("tests/test_data/test_data.csv")
    assert isinstance(data, pd.DataFrame)


def test_dataset_split_return() -> None:
    """Tests the returned object of the dataset_split function"""
    features, target = dataset_split("tests/test_data/test_data.csv")
    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, pd.Series)


def test_error_for_invalid_test_split_ratio() -> None:
    """Tests fails when test split ratio is greater than 1 in dataset_train_test_split function"""
    file_path = "tests/test_data/test_data.csv"
    random_state = 42
    test_split_ratio = 24
    with pytest.raises(ValueError):
        dataset_train_test_split(file_path, random_state, test_split_ratio)


def test_dataset_train_test_split_return() -> None:
    """Tests the returned object of the dataset_train_test_split function"""
    features_train, features_val, target_train, target_val = dataset_train_test_split(
        file_path="tests/test_data/test_data.csv",
        random_state=42,
        test_split_ratio=0.2
    )
    assert isinstance(features_train, pd.DataFrame)
    assert isinstance(features_val, pd.DataFrame)
    assert isinstance(target_train, pd.Series)
    assert isinstance(target_val, pd.Series)