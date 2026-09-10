from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from .config import FAILURE_MODE_COLUMNS, RANDOM_STATE


def permutation_importance_table(final_model: Pipeline, feature_data_test: pd.DataFrame, target_data_test:pd.Series,
                                 n_repeats: int=20, random_state: int=RANDOM_STATE) -> pd.DataFrame:
    result = permutation_importance(final_model, feature_data_test, target_data_test, scoring="average_precision",
                                    n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    return pd.DataFrame({"Feature": feature_data_test.columns, "Importance Mean": result.importances_mean,
                         "Importance Std": result.importances_std}).sort_values("Importance Mean", ascending=True)

def native_importance_table(final_model: Pipeline) -> pd.DataFrame:
    classifier = final_model.named_steps["classifier"]
    feature_names = final_model.named_steps["preprocessor"].get_feature_names_out()
    return pd.DataFrame({"Feature": feature_names, "Importance": classifier.feature_importances_}).sort_values("Importance", ascending=True)

def slice_performance(feature_data_test: pd.DataFrame, target_data_test: pd.Series, test_prediction: pd.Series,
                      test_probability: pd.Series, slice_column: str="Type") -> pd.DataFrame:
    rows = []
    for value in sorted(feature_data_test[slice_column].unique()):
        mask = (feature_data_test[slice_column] == value).to_numpy()
        y_slice = target_data_test[mask]
        rows.append({slice_column: value,"n": int(mask.sum()), "Failure rate": y_slice.mean(),
                     "Recall": recall_score(y_slice, test_prediction[mask], zero_division=0),
                     "Precision": precision_score(y_slice, test_prediction[mask], zero_division=0),
                     "PR-AUC": average_precision_score(y_slice, test_probability[mask]) if y_slice.nunique() > 1 else np.nan})
    return pd.DataFrame(rows)


def failure_mode_breakdown(maintenance_df: pd.DataFrame, feature_data_test: pd.DataFrame, test_prediction: pd.Series,
                           failure_mode_columns: Sequence[str]=FAILURE_MODE_COLUMNS) -> pd.DataFrame:
    failure_modes_test = maintenance_df.loc[feature_data_test.index, failure_mode_columns]
    rows = []
    for mode in failure_mode_columns:
        mode_mask = (failure_modes_test[mode] == 1).to_numpy()
        n_cases = int(mode_mask.sum())
        if n_cases == 0:
            continue
        caught = int(test_prediction[mode_mask].sum())
        rows.append({"Failure mode": mode, "Test cases": n_cases, "Caught (recall)": caught / n_cases})
    return pd.DataFrame(rows)

def build_error_analysis(feature_data_test: pd.DataFrame, target_data_test: pd.Series, test_probability: pd.Series, test_prediction: pd.Series) -> pd.DataFrame:
    error_analysis = feature_data_test.copy()
    error_analysis["Actual"] = target_data_test
    error_analysis["Probability"] = test_probability
    error_analysis["Predicted"] = test_prediction
    return error_analysis

def false_negatives_table(error_analysis: pd.DataFrame, maintenance_df: pd.DataFrame, failure_mode_columns: Sequence[str]=FAILURE_MODE_COLUMNS) -> pd.DataFrame:
    false_negatives = error_analysis[(error_analysis["Actual"] == 1) & (error_analysis["Predicted"] == 0)]
    false_negatives = false_negatives.sort_values("Probability", ascending=False)
    return false_negatives.join(maintenance_df.loc[false_negatives.index, failure_mode_columns])

def false_positives_table(error_analysis: pd.DataFrame) -> pd.DataFrame:
    false_positives = error_analysis[(error_analysis["Actual"] == 0) & (error_analysis["Predicted"] == 1)]
    return false_positives.sort_values("Probability", ascending=False)
