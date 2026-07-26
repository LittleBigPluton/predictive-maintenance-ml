import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from .config import RANDOM_STATE


def add_engineered_features(df):
    df = df.copy()
    rotational_speed_rad_s = df["Rotational speed"] * (2 * np.pi / 60)
    df["Power"] = df["Torque"] * rotational_speed_rad_s
    df["Temperature difference"] = df["Process temperature"] - df["Air temperature"]
    df["Torque x Tool wear"] = df["Torque"] * df["Tool wear"]
    return df


def get_engineered_numerical_features(numerical_features):
    return list(numerical_features) + ["Power", "Temperature difference", "Torque x Tool wear"]


def build_preprocessors(engineered_numerical_features, categorical_features):
    linear_preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), engineered_numerical_features), ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)])
    tree_preprocessor = ColumnTransformer(transformers=[("num", "passthrough", engineered_numerical_features), ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)])
    return linear_preprocessor, tree_preprocessor
