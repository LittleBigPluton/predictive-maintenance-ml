import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import classification_report


def create_output_directories(reports_dir=Path("reports"), artifacts_dir=Path("artifacts")):
    tables_dir = reports_dir / "cli" / "tables"
    figures_dir = reports_dir / "cli" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir, artifacts_dir


def save_table(table, output_path, include_index=True):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(table, pd.Series):
        table = table.to_frame()
    table.to_csv(output_path, index=include_index)


def classification_report_table(target_data_test, test_prediction):
    report = classification_report(target_data_test, test_prediction, target_names=["No failure", "Failure"], zero_division=0, output_dict=True)
    return pd.DataFrame(report).T


def confidence_interval_table(bootstrap_metrics, final_test_metrics):
    return pd.DataFrame({metric: {"Point estimate": final_test_metrics.get(metric, np.mean(values)), "Bootstrap mean": np.mean(values), "2.5th percentile": np.percentile(values, 2.5), "97.5th percentile": np.percentile(values, 97.5)} for metric, values in bootstrap_metrics.items() if values}).T


def save_model_artifacts(final_model, final_model_name, final_threshold, cost_minimizing_threshold, final_test_metrics, artifacts_dir, random_state, test_size):
    model_path = artifacts_dir / "final_model.joblib"
    metadata_path = artifacts_dir / "model_metadata.json"
    joblib.dump(final_model, model_path)
    metadata = {"model_name": final_model_name, "decision_threshold": float(final_threshold), "cost_minimizing_threshold": float(cost_minimizing_threshold), "random_state": random_state, "test_size": test_size, "test_metrics": final_test_metrics}
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=_json_default)
    return model_path, metadata_path


def print_final_summary(final_model_name, final_threshold, final_test_metrics, ci_table):
    print(f"Final model: {final_model_name}\n"
          f"Decision threshold: {final_threshold:.3f}\n"
          f"Test Average Precision: {final_test_metrics['PR-AUC']:.3f} "
          f"(95% CI [{ci_table.loc['PR-AUC', '2.5th percentile']:.3f}, {ci_table.loc['PR-AUC', '97.5th percentile']:.3f}])\n"
          f"Test Recall: {final_test_metrics['Recall']:.3f} "
          f"(95% CI [{ci_table.loc['Recall', '2.5th percentile']:.3f}, {ci_table.loc['Recall', '97.5th percentile']:.3f}])\n"
          f"Test Precision: {final_test_metrics['Precision']:.3f}")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")
