import pandas as pd


def missing_values_report(maintenance_df):
    return pd.DataFrame({"na_count": maintenance_df.isna().sum(), "null_count": maintenance_df.isnull().sum()})


def duplicate_check(maintenance_df, feature_data):
    n_duplicate_rows = int(feature_data.duplicated().sum())
    n_duplicate_full_rows = int(maintenance_df.drop(columns=["UDI", "Product ID"], errors="ignore").duplicated().sum())
    return {"duplicate_feature_rows": n_duplicate_rows, "duplicate_full_rows": n_duplicate_full_rows}


def class_distribution(target_data):
    return pd.DataFrame({"Count": target_data.value_counts(), "Percentage": target_data.value_counts(normalize=True) * 100})


def range_validation(feature_data, numerical_features):
    return pd.DataFrame({"min": feature_data[numerical_features].min(), "max": feature_data[numerical_features].max(), "negative_values": (feature_data[numerical_features] < 0).sum()})


def process_temperature_check(feature_data):
    return int((feature_data["Process temperature"] < feature_data["Air temperature"]).sum())
