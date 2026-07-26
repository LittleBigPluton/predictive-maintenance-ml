import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    make_scorer
)

from .config import (
    N_SPLITS,
    RANDOM_STATE,
    MIN_RECALL,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE
)

CV_SCORING = {"average_precision": "average_precision", "roc_auc": "roc_auc", "precision": "precision", "recall": "recall", "f1": "f1", "f2": make_scorer(fbeta_score, beta=2, zero_division=0)}

def make_cv(random_state=RANDOM_STATE, n_splits=N_SPLITS):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cross_validate_models(candidate_models, feature_data_train, target_data_train, cv):
    rows = []
    for model_name, model_pipeline in candidate_models.items():
        scores = cross_validate(model_pipeline, feature_data_train, target_data_train, cv=cv, scoring=CV_SCORING, n_jobs=-1)
        rows.append({"Model": model_name,"CV Average Precision": scores["test_average_precision"].mean(),"CV Average Precision Std": scores["test_average_precision"].std(),"CV ROC-AUC": scores["test_roc_auc"].mean(),
                                         "CV Precision": scores["test_precision"].mean(),"CV Recall": scores["test_recall"].mean(),"CV F1": scores["test_f1"].mean(),"CV F2": scores["test_f2"].mean()})

    return (pd.DataFrame(rows).sort_values("CV Average Precision",ascending=False).reset_index(drop=True))

def get_oof_probabilities(final_model, feature_data_train, target_data_train, cv):
    return cross_val_predict(final_model, feature_data_train, target_data_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def select_threshold(target_data_train, cv_probabilities, min_recall=MIN_RECALL):
    precision, recall, thresholds = precision_recall_curve(target_data_train, cv_probabilities)
    threshold_results = pd.DataFrame({"Threshold": thresholds, "Precision": precision[:-1], "Recall": recall[:-1]})
    candidates = threshold_results[threshold_results["Recall"] >= min_recall]
    selected_row = candidates.sort_values("Precision", ascending=False).iloc[0]
    return selected_row["Threshold"], threshold_results, (precision, recall, thresholds)


def get_calibration_curve(target_data_train, cv_probabilities, n_bins=10):
    return calibration_curve(target_data_train, cv_probabilities, n_bins=n_bins, strategy="quantile")


def expected_cost_curve(target_data_train, cv_probabilities, thresholds, cost_fn=COST_FALSE_NEGATIVE, cost_fp=COST_FALSE_POSITIVE):
    rows = []
    for threshold in np.unique(thresholds):
        predictions = (cv_probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(target_data_train, predictions).ravel()
        rows.append({"Threshold": threshold, "False Negatives": fn, "False Positives": fp, "Expected Cost": fn * cost_fn + fp * cost_fp})
    cost_curve = pd.DataFrame(rows)
    cost_minimizing_threshold = cost_curve.loc[cost_curve["Expected Cost"].idxmin(), "Threshold"]
    return cost_curve, cost_minimizing_threshold


def evaluate_on_test(final_model, final_model_name, final_threshold, feature_data_test, target_data_test):
    """The single, one-time evaluation of the final model on the held-out test set."""
    test_probability = final_model.predict_proba(feature_data_test)[:, 1]
    test_prediction = (test_probability >= final_threshold).astype(int)
    metrics = {"Model": final_model_name, "Threshold": final_threshold, "Accuracy": accuracy_score(target_data_test, test_prediction), "Balanced Accuracy": balanced_accuracy_score(target_data_test, test_prediction),
               "Precision": precision_score(target_data_test, test_prediction, zero_division=0), "Recall": recall_score(target_data_test, test_prediction, zero_division=0),
               "F1-score": f1_score(target_data_test, test_prediction, zero_division=0), "F2-score": fbeta_score(target_data_test, test_prediction, beta=2, zero_division=0),
               "ROC-AUC": roc_auc_score(target_data_test, test_probability), "PR-AUC": average_precision_score(target_data_test, test_probability),}
    return test_probability, test_prediction, metrics


def bootstrap_confidence_intervals(target_data_test, test_probability, final_threshold, n_bootstrap=1000, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    target_test_array = np.asarray(target_data_test)
    test_probability_array = np.asarray(test_probability)
    n_test = len(target_test_array)
    bootstrap_metrics = {"PR-AUC": [], "Recall": [], "Precision": [], "F1-score": []}
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n_test, n_test)
        y_sample = target_test_array[sample_idx]
        if y_sample.sum() == 0 or y_sample.sum() == len(y_sample):
            continue  # skip degenerate resamples with only one class present
        prob_sample = test_probability_array[sample_idx]
        pred_sample = (prob_sample >= final_threshold).astype(int)
        bootstrap_metrics["PR-AUC"].append(average_precision_score(y_sample, prob_sample))
        bootstrap_metrics["Recall"].append(recall_score(y_sample, pred_sample, zero_division=0))
        bootstrap_metrics["Precision"].append(precision_score(y_sample, pred_sample, zero_division=0))
        bootstrap_metrics["F1-score"].append(f1_score(y_sample, pred_sample, zero_division=0))
    return bootstrap_metrics
