import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_normalized_distribution(maintenance_df, feature, unit, target_column="Machine failure"):
    fig1, ax1 = plt.subplots()
    sns.histplot(data=maintenance_df, x=feature, hue=target_column, stat="density", common_norm=False, kde=True, ax=ax1)
    ax1.set_title(f"{feature} Distribution by {target_column} (normalized)")
    ax1.set_xlabel(f"{feature} [{unit}]")

    fig2, ax2 = plt.subplots()
    sns.boxplot(y=feature, x=target_column, data=maintenance_df, ax=ax2)
    ax2.set_title(f"{feature} Spread by {target_column}")
    return fig1, fig2


def plot_failure_rate_by_type(maintenance_df, target_column="Machine failure"):
    failure_rate_by_type = (maintenance_df.groupby("Type")[target_column].mean() * 100).reindex(["L", "M", "H"])
    fig, ax = plt.subplots()
    failure_rate_by_type.plot.bar(ax=ax, color=["steelblue", "goldenrod", "tomato"])
    ax.bar_label(ax.containers[0], fmt="%.2f%%")
    ax.set_title("Failure Rate by Product Type")
    ax.set_ylabel("Failure Rate (%)")
    return fig, failure_rate_by_type


def plot_torque_vs_speed(maintenance_df, target_column="Machine failure"):
    fig, ax = plt.subplots()
    sns.scatterplot(data=maintenance_df, x="Rotational speed", y="Torque", hue=target_column, alpha=0.6, ax=ax)
    ax.set_title("Torque vs. Rotational Speed by Machine Failure")
    return fig


def plot_power_estimate(maintenance_df, target_column="Machine failure"):
    rotational_speed_rad_s = maintenance_df["Rotational speed"] * (2 * np.pi / 60)
    power_estimate = maintenance_df["Torque"] * rotational_speed_rad_s
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(x=power_estimate, hue=maintenance_df[target_column], stat="density", common_norm=False, kde=True, ax=ax)
    ax.axvline(3500, color="black", linestyle="--", label="PWF band [3500, 9000] W")
    ax.axvline(9000, color="black", linestyle="--")
    ax.set_xlabel("Torque x Rotational speed [W]")
    ax.set_title("Estimated Power by Machine Failure")
    ax.legend()
    return fig


def plot_temperature_gap_vs_speed(maintenance_df, target_column="Machine failure"):
    temperature_difference = maintenance_df["Process temperature"] - maintenance_df["Air temperature"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=temperature_difference, y=maintenance_df["Rotational speed"], hue=maintenance_df[target_column],alpha=0.6, ax=ax)
    ax.axvline(8.6, color="black", linestyle="--", label="HDF threshold: diff < 8.6K")
    ax.axhline(1380, color="gray", linestyle="--", label="HDF threshold: speed < 1380 rpm")
    ax.set_xlabel("Process temperature - Air temperature [K]")
    ax.set_ylabel("Rotational speed [rpm]")
    ax.set_title("Temperature Gap vs. Rotational Speed by Machine Failure")
    ax.legend()
    return fig
