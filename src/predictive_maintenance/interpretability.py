import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import recall_score, precision_score, average_precision_score
from .config import RANDOM_STATE, FAILURE_MODE_COLUMNS

def permutation_importance_table(final_model, feature_data_test, target_data_test, n_repeats=20, random_state=RANDOM_STATE):
    result = permutation_importance(final_model, feature_data_test, target_data_test, scoring="average_precision", n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    return pd.DataFrame({"Feature": feature_data_test.columns, "Importance Mean": result.importances_mean, "Importance Std": result.importances_std}).sort_values("Importance Mean", ascending=True)

def native_importance_table(final_model):
    classifier = final_model.named_steps["classifier"]
    feature_names = final_model.named_steps["preprocessor"].get_feature_names_out()
    return pd.DataFrame({"Feature": feature_names, "Importance": classifier.feature_importances_}).sort_values("Importance", ascending=True)

def slice_performance(feature_data_test, target_data_test, test_prediction, test_probability, slice_column="Type"):
    rows = []
    for value in sorted(feature_data_test[slice_column].unique()):
        mask = (feature_data_test[slice_column] == value).to_numpy()
        y_slice = target_data_test[mask]
        rows.append({slice_column: value,"n": int(mask.sum()), "Failure rate": y_slice.mean(), "Recall": recall_score(y_slice, test_prediction[mask], zero_division=0),
                    "Precision": precision_score(y_slice, test_prediction[mask], zero_division=0), "PR-AUC": average_precision_score(y_slice, test_probability[mask]) if y_slice.nunique() > 1 else np.nan})
    return pd.DataFrame(rows)


def failure_mode_breakdown(maintenance_df, feature_data_test, test_prediction, failure_mode_columns=FAILURE_MODE_COLUMNS):
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

def build_error_analysis(feature_data_test, target_data_test, test_probability, test_prediction):
    error_analysis = feature_data_test.copy()
    error_analysis["Actual"] = target_data_test
    error_analysis["Probability"] = test_probability
    error_analysis["Predicted"] = test_prediction
    return error_analysis

def false_negatives_table(error_analysis, maintenance_df, failure_mode_columns=FAILURE_MODE_COLUMNS):
    false_negatives = error_analysis[(error_analysis["Actual"] == 1) & (error_analysis["Predicted"] == 0)]
    false_negatives = false_negatives.sort_values("Probability", ascending=False)
    return false_negatives.join(maintenance_df.loc[false_negatives.index, failure_mode_columns])

def false_positives_table(error_analysis):
    false_positives = error_analysis[(error_analysis["Actual"] == 0) & (error_analysis["Predicted"] == 1)]
    return false_positives.sort_values("Probability", ascending=False)
