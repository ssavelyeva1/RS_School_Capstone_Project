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


## Additional information
The code was formatted with *black*:
![black](https://user-images.githubusercontent.com/38406698/166344614-cc0a8f48-cd54-48e6-b1d1-f11e35d3a54c.png)

The code was linted with *flake8*:
![flake8](https://user-images.githubusercontent.com/38406698/166345519-8e1b1387-3c63-41a1-bf70-c4e9cede3c2b.png)

The code was type-checked with *mypy*:
![mypy](https://user-images.githubusercontent.com/38406698/166372514-d48f2e36-e202-4045-bba6-959d7d47c669.png)


