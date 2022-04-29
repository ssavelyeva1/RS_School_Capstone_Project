from pandas_profiling import ProfileReport
from pathlib import Path
import click
from module_9_project.data import get_dataset


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
def get_eda_profile(dataset_path: Path):
    df = get_dataset(dataset_path)
    profile = ProfileReport(df=df)
    profile.to_file("pandas_profile_test.html")
