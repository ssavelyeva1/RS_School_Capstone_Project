from sklearn.pipeline import Pipeline
import pytest

from module_9_project.pipeline import create_pipeline


def test_pipeline_return() -> None:
    """Tests the returned object of the pipeline function"""
    pipeline = create_pipeline("random_forest", "standard", 2, "gini", 100, 42)
    assert isinstance(pipeline, Pipeline)
