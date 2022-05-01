from sklearn.pipeline import Pipeline
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def create_pipeline(model_name: str, scaler_type: str, min_samples_split: int, criterion: str, max_depth: int, random_state: int):
    scaler, regressor = None, None

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "min_max":
        scaler = MinMaxScaler()

    if model_name == "random_forest":
        regressor = ensemble.RandomForestClassifier(
            min_samples_split=min_samples_split, criterion=criterion, max_depth=max_depth, random_state=random_state
        )
    elif model_name == "extra_trees":
        regressor = ensemble.ExtraTreesClassifier(
            min_samples_split=min_samples_split, criterion=criterion, max_depth=max_depth, random_state=random_state
        )

    pipe_steps = [("sca", scaler), ("reg", regressor)]

    return Pipeline(steps=pipe_steps)
