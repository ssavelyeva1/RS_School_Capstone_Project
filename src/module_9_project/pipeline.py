from sklearn.pipeline import Pipeline
from sklearn import ensemble


def create_pipeline(min_samples_split: int, criterion: str, max_depth: int, random_state: int):
    pipe_steps = []
    pipe_steps.append(
        (
            "classifier",
            ensemble.RandomForestClassifier(
                min_samples_split=min_samples_split, criterion=criterion, max_depth=max_depth, random_state=random_state
            )
        )
    )
    return Pipeline(steps=pipe_steps)