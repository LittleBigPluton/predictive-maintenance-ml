from sklearn.model_selection import train_test_split
from predictive_maintenance.data import (
    get_features_and_target,
    get_schema,
    load_raw_dataset,
    split_train_test)
from predictive_maintenance.evaluation import (
    cross_validate_models,
    make_cv
)
from predictive_maintenance.models import (
    build_models,
    compute_scale_pos_weight,
    tune_random_forest,
    tune_xgboost
)
from predictive_maintenance.processing import (
    add_engineered_features,
    build_preprocessors,
    get_engineered_numerical_features
)


def test_model_selection_and_tuning():
    # Load dataset
    maintenance_df, dataset = load_raw_dataset()
    feature_columns, _, numerical_features, categorical_features = get_schema(dataset)

    # Remove identifier columns from modeling
    feature_columns = [column for column in feature_columns if column not in ["UDI", "Product ID"]]
    numerical_features = [column for column in numerical_features if column != "UDI"]
    categorical_features = [column for column in categorical_features if column != "Product ID"]
    feature_data, target_data = get_features_and_target(maintenance_df, feature_columns)

    # Add engineered features
    feature_data = add_engineered_features(feature_data)
    engineered_numerical_features = (get_engineered_numerical_features(numerical_features))
    linear_preprocessor, tree_preprocessor = (build_preprocessors(engineered_numerical_features, categorical_features))

    # Create train/test split
    feature_data_train, feature_data_test, target_data_train, target_data_test = split_train_test(feature_data, target_data)

    assert not feature_data_train.empty
    assert not feature_data_test.empty
    assert len(feature_data_train) == len(target_data_train)
    assert len(feature_data_test) == len(target_data_test)

    # Use a smaller stratified sample to keep the test fast
    feature_data_sample, _, target_data_sample, _ = train_test_split(feature_data_train, target_data_train, train_size=1000, random_state=42, stratify=target_data_train)

    # Build candidate models
    scale_pos_weight = compute_scale_pos_weight(target_data_sample)
    candidate_models = build_models(tree_preprocessor, linear_preprocessor, scale_pos_weight)

    assert "Dummy Baseline" in candidate_models
    assert "Logistic Regression" in candidate_models
    assert "Random Forest" in candidate_models
    assert "XGBoost" in candidate_models
    assert "Balanced XGBoost" in candidate_models

    # Run a small cross-validation comparison
    cv = make_cv(n_splits=2)
    models_for_cv = {"Dummy Baseline": candidate_models["Dummy Baseline"],"Logistic Regression": candidate_models["Logistic Regression"]}
    cv_results = cross_validate_models(models_for_cv, feature_data_sample, target_data_sample, cv)

    assert cv_results is not None
    assert not cv_results.empty
    assert len(cv_results) == 2
    assert "Model" in cv_results.columns

    # Run one tuning iteration for each model
    random_forest_search = tune_random_forest(candidate_models, feature_data_sample, target_data_sample, cv, n_iter=1)
    xgboost_search = tune_xgboost(candidate_models, feature_data_sample, target_data_sample, cv, n_iter=1)

    assert random_forest_search.best_estimator_ is not None
    assert xgboost_search.best_estimator_ is not None
    assert random_forest_search.best_score_ >= 0
    assert xgboost_search.best_score_ >= 0
