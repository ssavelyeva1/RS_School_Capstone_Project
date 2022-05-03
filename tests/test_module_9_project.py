from click.testing import CliRunner
import pytest
from sklearn.pipeline import Pipeline

from module_9_project.train import train_model
from module_9_project.pipeline import create_pipeline


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


# def test_error_for_invalid_test_split_ratio(
#     runner: CliRunner
# ) -> None:
#     """It fails when test split ratio is greater than 1."""
#     result = runner.invoke(
#         train_model,
#         [
#             "--test-split-ratio",
#             42,
#         ],
#     )
#     assert result.exit_code == 2
#     assert "Invalid value for '--test-split-ratio'" in result.output


def test_pipeline_return() -> None:
    """Tests the returned object of the pipeline function"""
    pipeline = create_pipeline("random_forest", "standard", 2, "gini", 100, 42)
    assert isinstance(pipeline, Pipeline)

