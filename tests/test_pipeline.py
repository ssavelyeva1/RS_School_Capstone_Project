from sklearn.pipeline import Pipeline
import pytest

from module_9_project.pipeline import create_pipeline


def test_pipeline_return() -> None:
    """Test checks the returned object of the pipeline function"""
    pipeline = create_pipeline("random_forest", "standard", 2, "gini", 100, 42)
    assert isinstance(pipeline, Pipeline)


def test_pipeline_invalid_model() -> None:
    """Test fails when model_name is not correct in pipeline function"""
    with pytest.raises(ValueError):
        create_pipeline("invalid_model", "standard", 2, "gini", 100, 42)


def test_pipeline_invalid_scaler() -> None:
    """Test fails when scaler_type is not correct in pipeline function"""
    with pytest.raises(ValueError):
        create_pipeline("random_forest", "invalid_scaler", 2, "gini", 100, 42)
