import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from matplotlib.figure import Figure
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay
)


def save_figure(figure,output_path, * ,dpi = 300):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path,dpi=dpi,bbox_inches="tight")
    plt.close(figure)

def plot_cv_comparison(cv_results):
    ordered_results = cv_results.sort_values("CV Average Precision", ascending=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.barh(ordered_results["Model"], ordered_results["CV Average Precision"], xerr=ordered_results["CV Average Precision Std"], capsize=3)
    axis.set_xlabel("Cross-validated Average Precision")
    axis.set_title("Baseline Model Comparison")
    figure.tight_layout()
    return figure

def plot_threshold_tradeoff(thresholds, precision, recall, selected_threshold):
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(thresholds, precision[:-1], label="Precision")
    axis.plot(thresholds, recall[:-1], label="Recall")
    axis.axvline(selected_threshold, linestyle="--", label=f"Selected threshold = {selected_threshold:.3f}")
    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("Score")
    axis.set_title("Precision and Recall by Threshold ""(Training-Set OOF Predictions)")
    axis.legend()
    figure.tight_layout()
    return figure

def plot_calibration(mean_predicted_value, fraction_of_positives, model_name):
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(mean_predicted_value, fraction_of_positives, marker="o", label=model_name)
    axis.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed failure rate")
    axis.set_title("Calibration Curve (Training-Set OOF)")
    axis.legend()
    figure.tight_layout()
    return figure

def plot_expected_cost(cost_curve, selected_threshold, cost_minimizing_threshold):
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(cost_curve["Threshold"], cost_curve["Expected Cost"])
    axis.axvline(selected_threshold, linestyle="--", label=f"Recall-based = {selected_threshold:.3f}", color="red")
    axis.axvline(cost_minimizing_threshold, linestyle="--", label=f"Cost-minimizing = {cost_minimizing_threshold:.3f}")
    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("Expected cost")
    axis.set_title("Expected Cost by Decision Threshold")
    axis.legend()
    figure.tight_layout()
    return figure

def plot_confusion_matrix(target_data_test, test_prediction, model_name):
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(target_data_test, test_prediction, display_labels=["No failure", "Failure"], ax=axis)
    axis.set_title(f"{model_name} — Test Confusion Matrix")
    figure.tight_layout()
    return figure

def plot_roc_curve(target_data_test, test_probability, model_name):
    figure, axis = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(target_data_test, test_probability, name=model_name, ax=axis)
    axis.plot([0, 1], [0, 1], linestyle="--")
    axis.set_title(f"{model_name} — Test ROC Curve")
    figure.tight_layout()
    return figure

def plot_precision_recall_curve(target_data_test, test_probability, model_name):
    figure, axis = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(target_data_test, test_probability, name=model_name, ax=axis)
    axis.set_title(f"{model_name} — Test Precision-Recall Curve")
    figure.tight_layout()
    return figure

def plot_permutation_importance(importance_table, model_name):
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(importance_table["Feature"], importance_table["Importance Mean"], xerr=importance_table["Importance Std"])
    axis.set_xlabel("Decrease in Average Precision")
    axis.set_title(f"Permutation Importance — {model_name}")
    figure.tight_layout()
    return figure

def plot_native_importance(importance_table, model_name):
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(importance_table["Feature"], importance_table["Importance"])
    axis.set_xlabel("Importance")
    axis.set_title(f"Native Feature Importance — {model_name}")
    figure.tight_layout()
    return figure
