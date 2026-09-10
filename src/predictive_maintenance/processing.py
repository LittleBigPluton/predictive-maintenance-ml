import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rotational_speed_rad_s = df["Rotational speed"] * (2 * np.pi / 60)
    df["Power"] = df["Torque"] * rotational_speed_rad_s
    df["Temperature difference"] = df["Process temperature"] - df["Air temperature"]
    df["Torque x Tool wear"] = df["Torque"] * df["Tool wear"]
    return df

def get_engineered_numerical_features(numerical_features: list[str]) -> list[str]:
    return list(numerical_features) + ["Power", "Temperature difference", "Torque x Tool wear"]

def build_preprocessors(engineered_numerical_features: list[str], categorical_features: list[str]) -> tuple[ColumnTransformer, ColumnTransformer]:
    linear_preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), engineered_numerical_features),
                                                          ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)])
    tree_preprocessor = ColumnTransformer(transformers=[("num", "passthrough", engineered_numerical_features),
                                                        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)])
    return linear_preprocessor, tree_preprocessor
