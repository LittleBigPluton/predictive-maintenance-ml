
import numpy as np
import pandas as pd

from .data import (
    load_raw_dataset,
    get_schema,
    get_features_and_target,
    split_train_test
)
from .processing import (
    add_engineered_features,
    get_engineered_numerical_features,
    build_preprocessors
)
from .config import (
    RANDOM_STATE,
    TEST_SIZE
)
from .evaluation import (
    make_cv,
    cross_validate_models,
    get_oof_probabilities,
    select_threshold,
    get_calibration_curve,
    expected_cost_curve,
    evaluate_on_test,
    bootstrap_confidence_intervals
)
from .models import (
    build_models,
    compute_scale_pos_weight,
    tune_random_forest,
    tune_xgboost,
    select_final_model
)
from .interpretability import(
    permutation_importance_table,
    native_importance_table,
    slice_performance,
    failure_mode_breakdown,
    build_error_analysis,
    false_negatives_table,
    false_positives_table
)
from .plotting import (
    plot_calibration,
    plot_confusion_matrix,
    plot_cv_comparison,
    plot_expected_cost,
    plot_native_importance,
    plot_permutation_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_threshold_tradeoff,
    save_figure,
)
from .reporting import (
    create_output_directories,
    save_table,
    classification_report_table,
    confidence_interval_table,
    save_model_artifacts,
    print_final_summary
)

def main():
    # Configuration
    print(f"RANDOM_STATE = {RANDOM_STATE}")
    print(f"TEST_SIZE = {TEST_SIZE}")
    tables_dir, figures_dir, artifacts_dir = create_output_directories()

    # Data load
    maintenance_df, dataset = load_raw_dataset()
    identifier_columns = {"UDI", "Product ID"}
    feature_columns, _, numerical_features, categorical_features = get_schema(dataset)
    feature_columns = [column for column in feature_columns if column not in identifier_columns]
    numerical_features = [column for column in numerical_features if column not in identifier_columns ]
    categorical_features = [column for column in categorical_features if column not in identifier_columns]
    feature_data, target_data = get_features_and_target(maintenance_df, feature_columns)

    print(f"Shape of feature data: {feature_data.shape}")
    print(f"Shape of target data: {target_data.shape}")

    # Feature engineering
    ## Train/split
    feature_data_train, feature_data_test, target_data_train, target_data_test = split_train_test(feature_data, target_data)
    feature_data_train = add_engineered_features(feature_data_train)
    feature_data_test = add_engineered_features(feature_data_test)
    engineered_numerical_features = get_engineered_numerical_features(numerical_features)

    # Cross validation
    cv = make_cv()
    linear_preprocessor, tree_preprocessor = build_preprocessors(engineered_numerical_features, categorical_features)
    scale_pos_weight = compute_scale_pos_weight(target_data_train)
    print(f"scale_pos_weight (train only): {scale_pos_weight:.2f}")
    candidate_models = build_models(tree_preprocessor, linear_preprocessor, scale_pos_weight)

    # Baseline mode comparison
    cv_results = cross_validate_models(candidate_models, feature_data_train, target_data_train, cv)
    save_table(cv_results, tables_dir / "cross_validation_results.csv", include_index=False)

    ## Mean and standard deviation
    cv_figure = plot_cv_comparison(cv_results)
    save_figure(cv_figure, figures_dir / "cross_validation_comparison.png")

    # Hyperparameter Tuning
    ## Random Forest Tuning
    rf_search = tune_random_forest(candidate_models, feature_data_train, target_data_train, cv)
    print(f"Best Random Forest parameters: {rf_search.best_params_}")
    print(f"Best Random Forest CV Average Precision: {rf_search.best_score_:.4f}")

    ## XGBoost Tuning
    xgb_search = tune_xgboost(candidate_models, feature_data_train, target_data_train, cv)
    print("Best XGBoost parameters:")
    print(xgb_search.best_params_)
    print("\nBest CV average precision:")
    print(xgb_search.best_score_)

    ## Tuned vs untuned comparison
    baseline_scores = cv_results.set_index("Model")["CV Average Precision"]

    tuning_comparison = pd.DataFrame([{"Model": "Random Forest", "Untuned CV Average Precision": baseline_scores["Random Forest"], "Tuned CV Average Precision": rf_search.best_score_},
                                      {"Model": "XGBoost", "Untuned CV Average Precision": baseline_scores["XGBoost"], "Tuned CV Average Precision": xgb_search.best_score_}])
    tuning_comparison["Improvement"] = tuning_comparison["Tuned CV Average Precision"] - tuning_comparison["Untuned CV Average Precision"]
    print(tuning_comparison.round(4))
    save_table(tuning_comparison, tables_dir / "tuning_comparison.csv", include_index=False)

    ## Select Final model
    final_model_name, final_model = select_final_model({"Tuned Random Forest": rf_search, "Tuned XGBoost": xgb_search})
    print(f"Selected final model by CV Average Precision: {final_model_name}")

    # Threshold and Calibration
    ## Validation (OOF) threshold selection
    cv_probabilities = get_oof_probabilities(final_model, feature_data_train, target_data_train, cv)
    final_threshold, threshold_results, (precision, recall, thresholds) = select_threshold(target_data_train, cv_probabilities)
    print(f"Selected threshold (from training-set OOF predictions only): {final_threshold:.4f}")
    save_table(threshold_results, tables_dir / "threshold_results.csv", include_index=False)
    threshold_figure = plot_threshold_tradeoff(thresholds, precision, recall, final_threshold)
    save_figure(threshold_figure, figures_dir / "threshold_tradeoff.png")

    # Calibration curve
    fraction_of_positives, mean_predicted_value = get_calibration_curve(target_data_train, cv_probabilities)
    calibration_figure = plot_calibration(mean_predicted_value, fraction_of_positives, final_model_name)
    save_figure(calibration_figure, figures_dir / "calibration_curve.png")

    # Business-cost trade off
    cost_curve, cost_minimizing_threshold = expected_cost_curve(target_data_train, cv_probabilities, thresholds)
    cost_figure = plot_expected_cost(cost_curve, final_threshold, cost_minimizing_threshold)
    save_figure(cost_figure, figures_dir / "expected_cost_curve.png")
    save_table(cost_curve, tables_dir / "expected_cost_curve.csv", include_index=False)
    print(f"Cost-minimizing threshold: {cost_minimizing_threshold:.4f}")

    # Final test evaluation
    print(f"Final model: {final_model_name}")
    print(f"Final decision threshold: {final_threshold:.4f}")
    test_probability, test_prediction, final_test_metrics = evaluate_on_test(final_model, final_model_name, final_threshold, feature_data_test, target_data_test)
    save_table(pd.DataFrame([final_test_metrics]), tables_dir / "final_test_metrics.csv", include_index=False)
    classification_table = classification_report_table(target_data_test, test_prediction)
    print(classification_table.round(3))
    save_table(classification_table, tables_dir / "classification_report.csv")

    # Confusion matrix
    save_figure(plot_confusion_matrix(target_data_test,test_prediction,final_model_name),figures_dir / "confusion_matrix.png")
    save_figure(plot_roc_curve(target_data_test,test_probability,final_model_name),figures_dir / "roc_curve.png")
    save_figure(plot_precision_recall_curve(target_data_test,test_probability,final_model_name),figures_dir / "precision_recall_curve.png")

    # Confidence intervals
    bootstrap_metrics = bootstrap_confidence_intervals(target_data_test, test_probability, final_threshold)

    ci_table = pd.DataFrame({metric: {"Point estimate": final_test_metrics.get(metric, np.mean(values)),"Bootstrap mean": np.mean(values),"2.5th percentile": np.percentile(values, 2.5),
                                      "97.5th percentile": np.percentile(values, 97.5)}for metric, values in bootstrap_metrics.items()}).T
    print(ci_table.round(3))

    # Interpretability and Error Analysis
    ## Permutation importance
    permutation_importance_df = permutation_importance_table(final_model, feature_data_test, target_data_test)
    save_figure(plot_permutation_importance(permutation_importance_df,final_model_name),figures_dir / "permutation_importance.png")

    ## Native feature importance
    native_importance = native_importance_table(final_model)
    save_figure(plot_native_importance(native_importance,final_model_name),figures_dir / "native_importance.png")

    ## Slice performance
    slice_performances = slice_performance(feature_data_test, target_data_test, test_prediction, test_probability)
    save_table(slice_performances, tables_dir / "slice_performance.csv", include_index=False)

    ## Failure mode detection
    failure_mode_breakdowns = failure_mode_breakdown(maintenance_df, feature_data_test, test_prediction)
    save_table(failure_mode_breakdowns, tables_dir / "failure_mode_breakdown.csv", include_index=False)

    ## False negative Analysis
    error_analysis = build_error_analysis(feature_data_test, target_data_test, test_probability, test_prediction)
    false_negatives = false_negatives_table(error_analysis, maintenance_df)
    save_table(false_negatives, tables_dir / "false_negatives.csv")
    false_positives = false_positives_table(error_analysis)
    save_table(false_positives, tables_dir / "false_positives.csv")

    # Conclusion
    ## Key results
    print_final_summary(final_model_name, final_threshold, final_test_metrics, ci_table)
    model_path, metadata_path = save_model_artifacts(final_model, final_model_name, final_threshold, cost_minimizing_threshold, final_test_metrics, artifacts_dir, RANDOM_STATE, TEST_SIZE)
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")

    return 0
if __name__ == "__main__":
    raise SystemExit(main())
