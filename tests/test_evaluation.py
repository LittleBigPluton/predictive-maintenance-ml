import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from predictive_maintenance.evaluation import (
    bootstrap_confidence_intervals,
    evaluate_on_test,
    expected_cost_curve,
    get_calibration_curve,
    get_oof_probabilities,
    make_cv,
    select_threshold
)

def test_threshold_and_evaluation_functions():
    # Create a small imbalanced classification dataset
    feature_data, target_data = make_classification(n_samples=500, n_features=6, n_informative=4, n_redundant=1, weights=[0.85, 0.15], random_state=42)
    feature_data_train,feature_data_test,target_data_train,target_data_test = train_test_split(feature_data, target_data, test_size=0.20, random_state=42, stratify=target_data)
    final_model = LogisticRegression(max_iter=1000, random_state=42)
    cv = make_cv(n_splits=3)

    # Generate out-of-fold probabilities
    cv_probabilities = get_oof_probabilities(final_model, feature_data_train, target_data_train, cv)

    assert cv_probabilities is not None
    assert len(cv_probabilities) == len(target_data_train)
    assert np.all((cv_probabilities >= 0) & (cv_probabilities <= 1))

    # Select a recall-constrained threshold
    selected_threshold, threshold_results, precision_recall_values = select_threshold(target_data_train, cv_probabilities, min_recall=0.70)

    assert 0 <= selected_threshold <= 1
    assert not threshold_results.empty
    assert "Threshold" in threshold_results.columns
    assert "Precision" in threshold_results.columns
    assert "Recall" in threshold_results.columns
    assert len(precision_recall_values) == 3

    # Calculate calibration curve
    observed_fraction, predicted_probability = get_calibration_curve(target_data_train,cv_probabilities,n_bins=5)

    assert len(observed_fraction) > 0
    assert len(observed_fraction) == len(predicted_probability)

    # Calculate expected cost at different thresholds
    cost_curve, cost_minimizing_threshold = expected_cost_curve(target_data_train, cv_probabilities, threshold_results["Threshold"].to_numpy(), cost_fn=10, cost_fp=1)

    assert not cost_curve.empty
    assert "Expected Cost" in cost_curve.columns
    assert 0 <= cost_minimizing_threshold <= 1

    # Fit the final model before evaluating the test set
    final_model.fit(feature_data_train, target_data_train)
    test_probability, test_prediction, test_metrics = evaluate_on_test(final_model, "Logistic Regression", selected_threshold, feature_data_test, target_data_test)

    assert len(test_probability) == len(target_data_test)
    assert len(test_prediction) == len(target_data_test)
    assert set(np.unique(test_prediction)).issubset({0, 1})

    expected_metrics = {"Model", "Threshold", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-score", "F2-score", "ROC-AUC", "PR-AUC"}

    assert expected_metrics.issubset(test_metrics)

    # Use fewer bootstrap iterations to keep the test fast
    bootstrap_metrics = bootstrap_confidence_intervals(target_data_test, test_probability, selected_threshold, n_bootstrap=50, random_state=42)

    assert "PR-AUC" in bootstrap_metrics
    assert "Recall" in bootstrap_metrics
    assert "Precision" in bootstrap_metrics
    assert "F1-score" in bootstrap_metrics
    assert len(bootstrap_metrics["PR-AUC"]) > 0
