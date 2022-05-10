from pathlib import Path
from click.testing import CliRunner
from joblib import load
import pytest
from sklearn.model_selection import GridSearchCV

from module_9_project.train import train_model


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


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
            "tests/test_data/test_data.csv",
            "--save-model-path",
            "tests/test_data/model.joblib"
        ],
    )
    saved_model = Path("tests/test_data/model.joblib")
    assert saved_model.exists()
    loaded_model = load(saved_model)
    assert isinstance(loaded_model, GridSearchCV)
    Path.unlink(saved_model)



