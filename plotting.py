"""Plotting functions for visualizing model performance.

This module provides functions to create performance plots, histograms, and comparison
visualizations for particle physics mass regression models. Includes functions for
aggregating predictions, creating dual histograms, and plotting model performance
across different mass ranges with error bars.
"""

from collections import defaultdict

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

COLORS = ["blue", "red", "indigo", "darkorange", "gold"]


def aggregate_model_performance_by_mx(model_performance_dict):
    """
    Aggregates model performance data by M_x(true) values, combining multiple
    data points with the same M_x value into single aggregated points using
    proper weighted statistics.

    Args:
        model_performance_dict (dict): Dictionary where each key is a model name and value is a performance data dictionary.
            Each model's data dictionary must contain these keys:
              - "M_x(true)": List of true values
              - "mean": List of mean predictions
              - "+1σ": List of upper bound predictions
              - "-1σ": List of lower bound predictions
              - "sample_count": List of sample counts for each prediction

    Returns:
        dict: Aggregated dictionary with same structure but combined values for duplicate M_x(true) entries
    """
    aggregated_dict = {}

    for model_name, model_data in model_performance_dict.items():
        # Group data by M_x(true) value
        mx_groups = defaultdict(
            lambda: {
                "means": [],
                "plus_sigma": [],
                "minus_sigma": [],
                "sample_counts": [],
            }
        )

        mx_true_values = model_data["M_x(true)"]
        means = model_data["mean"]
        plus_sigmas = model_data["+1σ"]
        minus_sigmas = model_data["-1σ"]
        sample_counts = model_data["sample_count"]

        for i in range(len(mx_true_values)):
            mx_val = mx_true_values[i]
            mx_groups[mx_val]["means"].append(means[i])
            mx_groups[mx_val]["plus_sigma"].append(plus_sigmas[i])
            mx_groups[mx_val]["minus_sigma"].append(minus_sigmas[i])
            mx_groups[mx_val]["sample_counts"].append(sample_counts[i])

        # Aggregate each group using proper weighted statistics
        aggregated_mx_true = []
        aggregated_means = []
        aggregated_plus_sigma = []
        aggregated_minus_sigma = []

        for mx_val, group_data in mx_groups.items():
            aggregated_mx_true.append(mx_val)

            means_array = np.array(group_data["means"])
            plus_sigma_array = np.array(group_data["plus_sigma"])
            minus_sigma_array = np.array(group_data["minus_sigma"])
            counts_array = np.array(group_data["sample_counts"])

            # Weighted average for means
            weighted_mean = np.average(means_array, weights=counts_array)
            aggregated_means.append(weighted_mean)

            # For sigma bounds, we need to properly combine variances
            # Extract std from +1σ values
            std_values = plus_sigma_array - means_array  # Extract std from +1σ values

            # Proper formula for combining standard deviations from multiple groups:
            # σ²_combined = Σ(n_i × (σ²_i + (μ_i - μ_combined)²)) / Σ(n_i)
            variance_contributions = counts_array * (
                std_values**2  # Individual variances
                + (means_array - weighted_mean)
                ** 2  # Squared deviations from combined mean
            )
            combined_variance = np.sum(variance_contributions) / np.sum(counts_array)
            combined_std = np.sqrt(combined_variance)

            aggregated_plus_sigma.append(weighted_mean + combined_std)
            aggregated_minus_sigma.append(weighted_mean - combined_std)

        aggregated_dict[model_name] = {
            "M_x(true)": aggregated_mx_true,
            "mean": aggregated_means,
            "+1σ": aggregated_plus_sigma,
            "-1σ": aggregated_minus_sigma,
        }

    return aggregated_dict


def aggregate_predictions_by_bucket(y_true, y_pred, bucket_width=10, min_samples=5):
    """
    Aggregates predictions into buckets for variance visualization.

    Args:
        y_true (np.array): True values
        y_pred (np.array): Predicted values
        bucket_width (float): Width of prediction buckets in GeV/c²
        min_samples (int): Minimum number of samples required in a bucket to include it

    Returns:
        dict: Dictionary containing:
            - "M_x(pred)": Array of bucket midpoints (x-axis values)
            - "mean": Array of mean true/pred ratios for each bucket
            - "+1σ": Array of mean + 1 standard deviation
            - "-1σ": Array of mean - 1 standard deviation
    """
    # Calculate the ratio
    ratio = y_true / y_pred

    # Determine bucket boundaries
    min_pred = np.floor(y_pred.min() / bucket_width) * bucket_width
    max_pred = np.ceil(y_pred.max() / bucket_width) * bucket_width
    bucket_edges = np.arange(min_pred, max_pred + bucket_width, bucket_width)

    # Assign each prediction to a bucket
    bucket_indices = np.digitize(y_pred, bucket_edges) - 1

    bucket_midpoints = []
    mean_ratios = []
    plus_one_sigma = []
    minus_one_sigma = []

    # Process each bucket
    for i in range(len(bucket_edges) - 1):
        mask = bucket_indices == i
        bucket_ratios = ratio[mask]

        if len(bucket_ratios) >= min_samples:
            midpoint = (bucket_edges[i] + bucket_edges[i + 1]) / 2
            mean_ratio = np.mean(bucket_ratios)
            std_ratio = np.std(bucket_ratios)

            bucket_midpoints.append(midpoint)
            mean_ratios.append(mean_ratio)
            plus_one_sigma.append(mean_ratio + std_ratio)
            minus_one_sigma.append(mean_ratio - std_ratio)

    return {
        "M_x(pred)": np.array(bucket_midpoints),
        "mean": np.array(mean_ratios),
        "+1σ": np.array(plus_one_sigma),
        "-1σ": np.array(minus_one_sigma),
    }


def create_1var_histogram_with_marker(
    data, data_label, marker, marker_label, title, x_label, filename
):
    """
    Generates a histogram visualization with 200 bins and includes several reference lines:
    - The main marker line (red dashed)
    - The mean value (blue dashed)
    - ±1 standard deviation ranges (light blue dotted)

    Args:
        data (array-like):
            The data to plot in the histogram
        data_label (str):
            Label for the data series in the legend
        marker (float):
            Position for the vertical marker line (red dashed line)
        marker_label (str):
            Label for the marker line in the legend
        title (str):
            Title for the histogram
        x_label (str):
            Label for the x-axis
        filename (str):
            Path where the figure will be saved

    Returns:
        None:
            The function saves the figure to the specified filename but doesn't return any value

    """
    plt.style.use("default")

    # Process data
    npData = np.array(data)

    max_val = npData.max()
    mean_val = np.mean(npData)
    std_val = np.std(npData)
    median_val = np.median(npData)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    n, bins, patches = ax.hist(
        npData, bins=200, range=(0, max_val), alpha=0.5, edgecolor="black", color="blue"
    )
    if marker is not None:
        ax.axvline(x=marker, color="red", linestyle="--", linewidth=2)
    ax.axvline(x=mean_val, color="blue", linestyle="--", linewidth=2)
    ax.axvline(x=mean_val + std_val, color="lightblue", linestyle=":", linewidth=2)
    ax.axvline(x=mean_val - std_val, color="lightblue", linestyle=":", linewidth=2)

    # Add labels and format axes
    ax.set_title(title, fontsize=12)

    ax.set_xlabel(x_label, fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))

    ax.set_ylabel("Frequency", fontsize=10)

    legend_elements = [
        # Histogram
        Patch(color="blue", label=data_label),
        Line2D([0], [0], color="blue", linestyle=":", label=f"Mean: {mean_val:.2e}"),
        Line2D(
            [0],
            [0],
            color="lightblue",
            linestyle=":",
            label=f"±1σ: {mean_val-std_val:.2e}, {mean_val+std_val:.2e}",
        ),
        Patch(color="none", label=f"Median: {median_val:.2e}"),
    ]
    if marker_label is not None:
        legend_elements = [
            # True value Marker
            Line2D(
                [0], [0], color="red", linestyle="--", label=f"{marker_label}: {marker}"
            )
        ] + legend_elements
    ax.legend(handles=legend_elements, fontsize=10)

    plt.tight_layout()

    # Save to file
    plt.savefig(filename)
    plt.close(fig)


def create_2var_histogram_with_marker(
    data1,
    data_label1,
    data2,
    data_label2,
    marker,
    marker_label,
    title,
    x_label,
    filename,
):
    """
    Generates a histogram comparing two datasets with a marker line and statistical references.

    Args:
        data1 (array-like):
            First dataset to plot (shown in blue)
        data_label1 (str):
            Label for the first dataset in the legend
        data2 (array-like):
            Second dataset to plot (shown in green)
        data_label2 (str):
            Label for the second dataset in the legend
        marker (float):
            Position for the vertical marker line (red dashed line)
        marker_label (str):
            Label for the marker line in the legend
        title (str):
            Title for the histogram
        x_label (str):
            Label for the x-axis
        filename (str):
            Path where the figure will be saved

    Returns:
        None:
            The function saves the figure to the specified filename but doesn't return any value
    """
    plt.style.use("default")

    # Process data
    npData1 = np.array(data1)
    npData2 = np.array(data2)
    max_val = max(npData1.max(), npData2.max())

    mean_val1 = np.mean(npData1)
    std_val1 = np.std(npData1)
    median_val1 = np.median(npData1)

    mean_val2 = np.mean(npData2)
    std_val2 = np.std(npData2)
    median_val2 = np.median(npData2)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # Plot both histograms with transparency
    n1, bins1, patches1 = ax.hist(
        npData1,
        bins=200,
        range=(0, max_val),
        alpha=0.5,
        edgecolor="black",
        color="blue",
    )
    n2, bins2, patches2 = ax.hist(
        npData2,
        bins=200,
        range=(0, max_val),
        alpha=0.5,
        edgecolor="black",
        color="green",
    )

    # Add vertical lines for marker and statistics
    if marker is not None:
        ax.axvline(x=marker, color="red", linestyle="--", linewidth=2)

    # Dataset 1 statistics (solid lines)
    ax.axvline(x=mean_val1, color="blue", linestyle="--", linewidth=2)
    ax.axvline(x=mean_val1 + std_val1, color="lightblue", linestyle=":", linewidth=2)
    ax.axvline(x=mean_val1 - std_val1, color="lightblue", linestyle=":", linewidth=2)

    # Dataset 2 statistics (dashed lines)
    ax.axvline(x=mean_val2, color="green", linestyle="--", linewidth=2)
    ax.axvline(x=mean_val2 + std_val2, color="lightgreen", linestyle=":", linewidth=2)
    ax.axvline(x=mean_val2 - std_val2, color="lightgreen", linestyle=":", linewidth=2)

    # Add labels and format axes
    ax.set_title(title, fontsize=12)

    ax.set_xlabel(x_label, fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))

    ax.set_ylabel("Frequency", fontsize=10)

    # Create legend with statistical information
    legend_elements = [
        # Histogram 1
        Patch(color="blue", alpha=0.5, label=f"{data_label1}"),
        Line2D([0], [0], color="blue", linestyle=":", label=f"Mean: {mean_val1:.2e}"),
        Line2D(
            [0],
            [0],
            color="lightblue",
            linestyle=":",
            label=f"±1σ: {mean_val1-std_val1:.2e}, {mean_val1+std_val1:.2e}",
        ),
        Patch(color="none", label=f"Median: {median_val1:.2e}"),
        # Histogram 2
        Patch(color="green", alpha=0.5, label=f"{data_label2}"),
        Line2D([0], [0], color="green", linestyle=":", label=f"Mean: {mean_val2:.2e}"),
        Line2D(
            [0],
            [0],
            color="lightgreen",
            linestyle=":",
            label=f"±1σ: {mean_val2-std_val2:.2e}, {mean_val2+std_val2:.2e}",
        ),
        Patch(color="none", label=f"Median: {median_val2:.2e}"),
    ]
    if marker_label is not None:
        # True value Marker
        legend_elements = [
            Line2D(
                [0], [0], color="red", linestyle="--", label=f"{marker_label}: {marker}"
            )
        ] + legend_elements

    ax.legend(handles=legend_elements, fontsize=10)

    plt.tight_layout()

    # Save to file
    plt.savefig(filename)
    plt.close(fig)


def compare_performance_all(model_performance_dict, filename, err_bar_h_spacing=2):
    """
    Generates an error bar plot comparing prediction accuracy of multiple models.

    Args:
        model_performance_dict (dict): Dictionary where each key is a model name and value is a performance data dictionary.
            Each model's data dictionary must contain these keys:
              - "M_x(true)": Array of true values
              - "mean": Array of mean predictions
              - "+1σ": Array of upper bound predictions
              - "-1σ": Array of lower bound predictions
        filename (str): Path where the figure will be saved

    Returns:
        None:
            The function saves the figure to the specified filename but doesn't return any value
    """
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # Plot each model with error bars
    for i, (model_name, model_data) in enumerate(model_performance_dict.items()):
        true_val = np.array(model_data["M_x(true)"])
        mean = np.array(model_data["mean"])
        plusOneSigma = np.array(model_data["+1σ"])
        minusOneSigma = np.array(model_data["-1σ"])

        model_color = COLORS[i]

        # Calculate ratios
        mean_ratio = mean / true_val
        plus_ratio = plusOneSigma / true_val
        minus_ratio = minusOneSigma / true_val

        # Calculate error bar sizes (asymmetric errors)
        upper_error = plus_ratio - mean_ratio
        lower_error = mean_ratio - minus_ratio

        # Stagger x-values to avoid overlapping error bars
        staggered_x = (
            true_val + err_bar_h_spacing * i
        )  # Increased spacing for error bars

        # Plot error bars
        ax.errorbar(
            staggered_x,
            mean_ratio,
            yerr=[lower_error, upper_error],
            fmt="o",
            color=model_color,
            capsize=3,
            capthick=1,
            elinewidth=1.5,
            markersize=4,
            alpha=0.8,
            label=model_name,
        )

    # Add a green horizontal line for the perfect prediction ratio = 1
    ax.axhline(
        y=1.0,
        color="green",
        linestyle="-",
        alpha=0.7,
        linewidth=2,
        label="Perfect prediction",
    )

    # Set labels and title
    ax.set_title("Model Prediction Accuracy Comparison", fontsize=14)

    ax.set_xlabel("M_x(true) [GeV/c²]", fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))

    ax.set_ylabel("M_x(pred) / M_x(true)", fontsize=12)

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(fontsize=10, loc="best")

    plt.tight_layout()

    plt.savefig(filename)
    plt.close(fig)


def compare_variance_all(model_performance_dict, filename, err_bar_h_spacing=2):
    """
    Generates an error bar plot comparing prediction variance of multiple models.
    X-axis shows predictions, Y-axis shows true/pred ratio to highlight variance.

    Args:
        model_performance_dict (dict): Dictionary where each key is a model name and value is a performance data dictionary.
            Each model's data dictionary must contain these keys:
              - "M_x(pred)": Array of prediction bucket midpoints
              - "mean": Array of mean true/pred ratios
              - "+1σ": Array of upper bound ratios
              - "-1σ": Array of lower bound ratios
        filename (str): Path where the figure will be saved

    Returns:
        None:
            The function saves the figure to the specified filename but doesn't return any value
    """
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # Plot each model with error bars
    for i, (model_name, model_data) in enumerate(model_performance_dict.items()):
        pred_val = np.array(model_data["M_x(pred)"])
        mean_ratio = np.array(model_data["mean"])
        plus_ratio = np.array(model_data["+1σ"])
        minus_ratio = np.array(model_data["-1σ"])

        model_color = COLORS[i]

        # Calculate error bar sizes (asymmetric errors)
        upper_error = plus_ratio - mean_ratio
        lower_error = mean_ratio - minus_ratio

        # Stagger x-values to avoid overlapping error bars
        staggered_x = (
            pred_val + err_bar_h_spacing * i
        )  # Increased spacing for error bars

        # Plot error bars
        ax.errorbar(
            staggered_x,
            mean_ratio,
            yerr=[lower_error, upper_error],
            fmt="o",
            color=model_color,
            capsize=3,
            capthick=1,
            elinewidth=1.5,
            markersize=4,
            alpha=0.8,
            label=model_name,
        )

    # Add a green horizontal line for the perfect prediction ratio = 1
    ax.axhline(
        y=1.0,
        color="green",
        linestyle="-",
        alpha=0.7,
        linewidth=2,
        label="Perfect prediction",
    )

    # Set labels and title
    ax.set_title("Model Prediction Variance Comparison", fontsize=14)

    ax.set_xlabel("M_x(pred) [GeV/c²]", fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))

    ax.set_ylabel("M_x(true) / M_x(pred)", fontsize=12)

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(fontsize=10, loc="best")

    plt.tight_layout()

    plt.savefig(filename)
    plt.close(fig)


def compare_performance_single(model_performance_dict, filename):
    """
    Generates a line plot comparing prediction accuracy of multiple models for a single value.

    Args:
        model_performance_dict (dict): Dictionary where each key is a model name and value is a performance data dictionary.
            Each model's data dictionary must contain these keys:
                - "mean": Float representing the mean prediction ratio
                - "+1σ": Float representing the upper bound prediction ratio
                - "-1σ": Float representing the lower bound prediction ratio
        filename (str): Path where the figure will be saved

    Returns:
        None: The function saves the figure to the specified filename
    """

    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Plot each model
    for i, (model_name, model_data) in enumerate(model_performance_dict.items()):
        true_val = model_data["M_x(true)"]
        mean = model_data["mean"]
        plusOneSigma = model_data["+1σ"]
        minusOneSigma = model_data["-1σ"]

        model_color = COLORS[i]

        ax.axhline(
            y=(mean / true_val),
            color=model_color,
            linestyle="-",
            alpha=0.9,
            label=model_name + " mean",
        )
        ax.axhline(
            y=(plusOneSigma / true_val),
            color=model_color,
            linestyle=":",
            alpha=0.65,
            label=model_name + " +1σ",
        )
        ax.axhline(
            y=(minusOneSigma / true_val),
            color=model_color,
            linestyle=":",
            alpha=0.65,
            label=model_name + " -1σ",
        )

    # Add a green diagonal line for the perfect prediction ratio = 1
    ax.axhline(y=1.0, color="green", linestyle="-", alpha=1.0)

    # Set labels and title
    ax.set_title("Model Prediction Accuracy Comparison", fontsize=12)

    ax.set_xlabel("M_x(true)", fontsize=10)

    ax.set_ylabel("M_x(pred) / M_x(true)", fontsize=10)

    # Add legend
    ax.legend(fontsize=10)

    plt.tight_layout()

    plt.savefig(filename)
    plt.close(fig)
