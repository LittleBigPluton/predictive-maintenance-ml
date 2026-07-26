import matplotlib.pyplot as plt

from predictive_maintenance.data import (
    get_features_and_target,
    get_schema,
    load_raw_dataset
)
from predictive_maintenance.data_quality import (
    class_distribution,
    duplicate_check,
    missing_values_report,
    process_temperature_check,
    range_validation
)
from predictive_maintenance.eda_visualization import (
    plot_failure_rate_by_type,
    plot_normalized_distribution,
    plot_power_estimate,
    plot_temperature_gap_vs_speed,
    plot_torque_vs_speed
)


def test_data_quality_functions():
    maintenance_df, dataset = load_raw_dataset()
    feature_columns, _, numerical_features, _ = get_schema(dataset)
    feature_data, target_data = get_features_and_target(maintenance_df, feature_columns)
    missing_report = missing_values_report(maintenance_df)

    assert missing_report is not None
    assert not missing_report.empty
    assert "na_count" in missing_report.columns
    assert "null_count" in missing_report.columns

    duplicate_report = duplicate_check(maintenance_df, feature_data)

    assert duplicate_report is not None
    assert "duplicate_feature_rows" in duplicate_report
    assert "duplicate_full_rows" in duplicate_report

    target_distribution = class_distribution(target_data)

    assert target_distribution is not None
    assert not target_distribution.empty
    assert "Count" in target_distribution.columns
    assert "Percentage" in target_distribution.columns

    range_report = range_validation(feature_data, numerical_features)

    assert range_report is not None
    assert not range_report.empty
    assert "min" in range_report.columns
    assert "max" in range_report.columns
    assert "negative_values" in range_report.columns

    invalid_temperature_rows = process_temperature_check(feature_data)

    assert isinstance(invalid_temperature_rows, int)
    assert invalid_temperature_rows >= 0


def test_eda_plotting_functions():
    maintenance_df, _ = load_raw_dataset()
    histogram_figure, boxplot_figure = (plot_normalized_distribution(maintenance_df, feature="Torque", unit="Nm"))

    assert histogram_figure is not None
    assert boxplot_figure is not None

    failure_rate_figure, failure_rates = (plot_failure_rate_by_type(maintenance_df))

    assert failure_rate_figure is not None
    assert not failure_rates.empty

    torque_speed_figure = plot_torque_vs_speed(maintenance_df)
    power_figure = plot_power_estimate(maintenance_df)
    temperature_figure = plot_temperature_gap_vs_speed(maintenance_df)

    assert torque_speed_figure is not None
    assert power_figure is not None
    assert temperature_figure is not None

    plt.close("all")
