import pytest
import pandas as pd
from pathlib import Path
from click.testing import CliRunner

from module_9_project.predict import predict_model


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_test_data_path(
    runner: CliRunner
) -> None:
    """Test fails when path for function predict_model is invalid."""
    result = runner.invoke(
        predict_model,
        [
            "--test-path",
            "invalid/path/test.csv"
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '-t' / '--test-path': File 'invalid/path/test.csv' does not exist." in result.output


def test_saved_predictions_for_predict_model(
        runner: CliRunner
) -> None:
    """Test checks saved predictions for predict_model."""
    runner.invoke(
        predict_model,
        [
            "--test-path",
            "tests/test_data/data_for_pred.csv",
            "--saved-model-path",
            "tests/test_data/model.joblib",
            "--prediction-path",
            "tests/test_data"
        ],
    )
    saved_preds = Path("tests/test_data/submit.csv")
    assert saved_preds.exists()
    loaded_preds = pd.read_csv(saved_preds)
    assert isinstance(loaded_preds, pd.DataFrame)
    assert loaded_preds.shape == (20, 2)
    Path.unlink(saved_preds)
