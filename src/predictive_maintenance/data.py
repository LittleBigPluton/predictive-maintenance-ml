import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from .config import TEST_SIZE, RANDOM_STATE

def load_raw_dataset():
    dataset = fetch_ucirepo(id=601)
    maintenance_df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    return maintenance_df, dataset


def get_schema(dataset):
    variables = dataset.variables
    feature_columns = variables.loc[variables["role"] == "Feature", "name"]
    target_columns = variables.loc[variables["role"] == "Target", "name"]
    numerical_features = variables.loc[(variables["type"].isin(["Integer", "Continuous"])) & (variables["role"] == "Feature"), "name"]
    categorical_features = variables.loc[(variables["type"] == "Categorical") & (variables["role"] == "Feature"), "name"]
    return feature_columns, target_columns, numerical_features, categorical_features


def get_features_and_target(maintenance_df, feature_columns, target_column="Machine failure"):
    feature_data = maintenance_df[feature_columns].copy()
    target_data = maintenance_df[target_column]
    return feature_data, target_data

def split_train_test(feature_data, target_data):
    return train_test_split(feature_data, target_data, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target_data)
