import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from predictive_maintenance.interpretability import (
    build_error_analysis,
    failure_mode_breakdown,
    false_negatives_table,
    false_positives_table,
    native_importance_table,
    permutation_importance_table,
    slice_performance
)


def test_interpretability_functions():
    # Create a small classification dataset
    numeric_data, target_data = make_classification(n_samples=120, n_features=3, n_informative=3, n_redundant=0, weights=[0.75, 0.25], random_state=42)
    feature_data = pd.DataFrame(numeric_data, columns=["Torque", "Rotational speed", "Tool wear"])
    feature_data["Type"] = np.resize(["L", "M", "H"], len(feature_data))
    target_data = pd.Series(target_data, name="Machine failure", index=feature_data.index)

    # Add failure modes for post-hoc analysis
    maintenance_df = feature_data.copy()
    maintenance_df["TWF"] = target_data
    maintenance_df["HDF"] = ((target_data == 1) & (maintenance_df.index % 2 == 0)).astype(int)
    feature_data_train, feature_data_test, target_data_train, target_data_test = train_test_split(feature_data, target_data, test_size=0.25, random_state=42, stratify=target_data)
    preprocessor = ColumnTransformer(transformers=[("num", "passthrough", ["Torque", "Rotational speed", "Tool wear"]), ("cat",OneHotEncoder(drop="first", handle_unknown="ignore"), ["Type"])])
    final_model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier",RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1))])
    final_model.fit(feature_data_train, target_data_train)
    test_probability = final_model.predict_proba(feature_data_test)[:, 1]

    # Create predictions with at least one false positive
    # and one false negative for error-table testing
    test_prediction = target_data_test.to_numpy().copy()
    positive_position = np.flatnonzero(target_data_test.to_numpy() == 1)[0]
    negative_position = np.flatnonzero(target_data_test.to_numpy() == 0)[0]
    test_prediction[positive_position] = 0
    test_prediction[negative_position] = 1

    # Permutation importance
    permutation_table = permutation_importance_table(final_model, feature_data_test, target_data_test, n_repeats=2)

    assert not permutation_table.empty
    assert "Feature" in permutation_table.columns
    assert "Importance Mean" in permutation_table.columns
    assert "Importance Std" in permutation_table.columns

    # Native tree-model importance
    native_table = native_importance_table(final_model)

    assert not native_table.empty
    assert "Feature" in native_table.columns
    assert "Importance" in native_table.columns

    # Product-type slice performance
    slice_table = slice_performance(feature_data_test, target_data_test, test_prediction, test_probability)

    assert not slice_table.empty
    assert "Type" in slice_table.columns
    assert "Recall" in slice_table.columns
    assert "Precision" in slice_table.columns
    assert "PR-AUC" in slice_table.columns

    # Failure-mode recall
    failure_table = failure_mode_breakdown(maintenance_df, feature_data_test, test_prediction, failure_mode_columns=["TWF", "HDF"])

    assert not failure_table.empty
    assert "Failure mode" in failure_table.columns
    assert "Test cases" in failure_table.columns
    assert "Caught (recall)" in failure_table.columns

    # Error-analysis table
    error_analysis = build_error_analysis(feature_data_test, target_data_test, test_probability, test_prediction)

    assert len(error_analysis) == len(feature_data_test)
    assert "Actual" in error_analysis.columns
    assert "Probability" in error_analysis.columns
    assert "Predicted" in error_analysis.columns

    # False-negative and false-positive tables
    false_negatives = false_negatives_table(error_analysis, maintenance_df, failure_mode_columns=["TWF", "HDF"])
    false_positives = false_positives_table(error_analysis)

    assert not false_negatives.empty
    assert not false_positives.empty
    assert "TWF" in false_negatives.columns
    assert "HDF" in false_negatives.columns
