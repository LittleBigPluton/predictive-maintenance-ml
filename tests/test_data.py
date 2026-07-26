from predictive_maintenance.data import (
    get_features_and_target,
    get_schema,
    load_raw_dataset
)
from predictive_maintenance.processing import (
    add_engineered_features,
    build_preprocessors,
    get_engineered_numerical_features
)


def test_data_and_processing():
    # Load the AI4I dataset
    maintenance_df, dataset = load_raw_dataset()

    assert maintenance_df is not None
    assert dataset is not None
    assert not maintenance_df.empty

    # Extract schema information
    feature_columns, target_columns, numerical_features, categorical_features = get_schema(dataset)

    assert len(feature_columns) > 0
    assert len(target_columns) > 0
    assert len(numerical_features) > 0
    assert len(categorical_features) > 0

    # Separate features and target
    feature_data, target_data = get_features_and_target(maintenance_df, feature_columns)

    assert not feature_data.empty
    assert not target_data.empty
    assert len(feature_data) == len(target_data)

    # Add engineered features
    engineered_data = add_engineered_features(feature_data)
    expected_features = ["Power", "Temperature difference", "Torque x Tool wear"]

    for feature in expected_features:
        assert feature in engineered_data.columns

    # Update the numerical feature list
    engineered_numerical_features = get_engineered_numerical_features(numerical_features)

    for feature in expected_features:
        assert feature in engineered_numerical_features

    # Build preprocessing pipelines
    linear_preprocessor, tree_preprocessor = build_preprocessors(engineered_numerical_features, categorical_features)

    assert linear_preprocessor is not None
    assert tree_preprocessor is not None
