The Capstone Project for the Rolling Scopes School Machine learning intro course.

This project uses [Forest Cover type](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset and allows you to train models and tune hyperparameters for predicting the forest cover type.

## Usage
To start using this package in practice you should follow these steps:
1. Clone this repository to your machine.
2. Download [Forest Cover type](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset, save train file in CSV format locally (default path is *data/train.csv* in repository's root).
3. Make sure Python and Poetry are installed on your machine (for these project Python 3.9.12 and Poetry 1.1.13 were used).
4. Install project dependencies by running the following command in a terminal from the project root:
```sh
poetry install --no-dev
```
5. To get the full EDA report in HTML format run eda via this command:
```sh
poetry run eda
```
6. Run train with the following command:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can choose the model (Random Forest or Extra Trees classifier), configure it's hyperparameters and feature engineering techniques in the command line interface. To get a full list of them, use help:
```sh
poetry run train --help
```
7. Run MLflow UI to see the information about experiments you conducted (like in the example below) run this command:
```sh
poetry run mlflow ui
```
![mlflow](https://user-images.githubusercontent.com/38406698/166170177-fd28496d-54ed-4aa9-a8d8-549285836fcb.png)

## Development
Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
To run the tests run the following command:
```
poetry run pytest
```
![pytest](https://user-images.githubusercontent.com/38406698/166552440-448f633a-1263-4c99-a0be-d4929f1c5569.png)
*nox* was used for multiple sessions run (including black formatting, flake8 linting and mypy type-checking):
![nox](https://user-images.githubusercontent.com/38406698/166551661-987043f4-da8e-40f9-9938-264f30680682.png)
To start nox before commiting use command:
```
poetry run nox
```


