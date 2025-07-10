"""
Module Name: loss_functions

Description:
    Plotting functions to visualize model performance

Usage:
Author:
Date:
License:
"""
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np


COLORS = ["blue", "red", "indigo", "darkorange", "gold"]


def create_1var_histogram_with_marker(data, data_label, marker, marker_label, title, x_label, filename):
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
    n, bins, patches = ax.hist(npData, bins=200, range=(0, max_val), 
                                 alpha=0.5, edgecolor='black', color="blue")
    ax.axvline(x=marker, color="red", linestyle="--", linewidth=2)
    ax.axvline(x=mean_val, color="blue", linestyle="--", linewidth=2)   
    ax.axvline(x=mean_val+std_val, color="lightblue", linestyle=":", linewidth=2)
    ax.axvline(x=mean_val-std_val, color="lightblue", linestyle=":", linewidth=2)    

    # Add labels and format axes
    ax.set_title(title, fontsize=12)

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_xticks(np.linspace(0, max_val, 6))
    ax.set_xticklabels([f'{x:.2e}' for x in np.linspace(0, max_val, 6)])

    ax.set_ylabel("Frequency", fontsize=10)
    
    legend_elements = [
        # True value Marker
        Line2D([0], [0], color="red", linestyle='--', label=f"{marker_label}: {marker}"),
        # Histogram
        Patch(color="blue", label=data_label),
        Line2D([0], [0], color="blue", linestyle=":", label=f"Mean: {mean_val:.2e}"),
        Line2D([0], [0], color="lightblue", linestyle=":", label=f"±1σ: {mean_val-std_val:.2e}, {mean_val+std_val:.2e}"),
        Patch(color="none", label=f"Median: {median_val:.2e}"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)

    plt.tight_layout()
    
    # Save to file
    plt.savefig(filename)
    plt.close(fig)


def create_2var_histogram_with_marker(data1, data_label1, data2, data_label2, marker, marker_label, 
                                      title, x_label, filename):
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
    plt.style.use('default')

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
    n1, bins1, patches1 = ax.hist(npData1, bins=200, range=(0, max_val), 
                                 alpha=0.5, edgecolor='black', color="blue")
    n2, bins2, patches2 = ax.hist(npData2, bins=200, range=(0, max_val), 
                                 alpha=0.5, edgecolor='black', color="green")
    
    # Add vertical lines for marker and statistics
    ax.axvline(x=marker, color='red', linestyle='--', linewidth=2)
    
    # Dataset 1 statistics (solid lines)
    ax.axvline(x=mean_val1, color='blue', linestyle='--', linewidth=2)
    ax.axvline(x=mean_val1+std_val1, color='lightblue', linestyle=':', linewidth=2)
    ax.axvline(x=mean_val1-std_val1, color='lightblue', linestyle=':', linewidth=2)
    
    # Dataset 2 statistics (dashed lines)
    ax.axvline(x=mean_val2, color='green', linestyle='--', linewidth=2)
    ax.axvline(x=mean_val2+std_val2, color='lightgreen', linestyle=':', linewidth=2)
    ax.axvline(x=mean_val2-std_val2, color='lightgreen', linestyle=':', linewidth=2)

    # Add labels and format axes
    ax.set_title(title, fontsize=12)

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_xticks(np.linspace(0, max_val, 6))
    ax.set_xticklabels([f"{x:.2e}" for x in np.linspace(0, max_val, 6)])

    ax.set_ylabel("Frequency", fontsize=10)
    
    # Create legend with statistical information
    legend_elements = [
        # True value Marker
        Line2D([0], [0], color="red", linestyle='--', label=f"{marker_label}: {marker}"),
        # Histogram 1
        Patch(color="blue", alpha=0.5, label=f"{data_label1}"),
        Line2D([0], [0], color="blue", linestyle=':', label=f"Mean: {mean_val1:.2e}"),
        Line2D([0], [0], color="lightblue", linestyle=':', label=f"±1σ: {mean_val1-std_val1:.2e}, {mean_val1+std_val1:.2e}"),
        Patch(color="none", label=f"Median: {median_val1:.2e}"),
        # Histogram 2
        Patch(color="green", alpha=0.5, label=f"{data_label2}"),
        Line2D([0], [0], color="green", linestyle=':', label=f"Mean: {mean_val2:.2e}"),
        Line2D([0], [0], color="lightgreen", linestyle=':', label=f"±1σ: {mean_val2-std_val2:.2e}, {mean_val2+std_val2:.2e}"),
        Patch(color="none", label=f"Median: {median_val2:.2e}")
    ]
    ax.legend(handles=legend_elements, fontsize=10)

    plt.tight_layout()
    
    # Save to file
    plt.savefig(filename)
    plt.close(fig)


def compare_performance_all(model_performance_dict, filename):
    """
    Generates a scatter plot comparing prediction accuracy of multiple models.
    
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
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Plot each model
    for i, (model_name, model_data) in enumerate(model_performance_dict.items()):
        true_val = np.array(model_data["M_x(true)"])
        mean = np.array(model_data["mean"])
        plusOneSigma = np.array(model_data["+1σ"])
        minusOneSigma = np.array(model_data["-1σ"])

        model_color = COLORS[i]
        
        # Stagger x-values to avoid overlapping points
        staggered_x = true_val + 2 * i

        ax.scatter(
            staggered_x, 
            (mean / true_val),
            color=model_color,
            alpha=0.6,
            s=5,
            label=model_name + " mean",
        )

        ax.scatter(
            staggered_x,
            (plusOneSigma / true_val),
            color=model_color,
            alpha=0.25,
            s=5,
            label=model_name + " +1σ",
        )

        ax.scatter(
            staggered_x,
            (minusOneSigma / true_val),
            color=model_color,
            alpha=0.2,
            s=5,
            label=model_name + " -1σ",
        )

    # Add a green diagonal line for the perfect prediction ratio = 1
    ax.axhline(y=1.0, color='green', linestyle='-', alpha=1.0)

    # Set labels and title
    ax.set_title("Model Prediction Accuracy Comparison", fontsize=12)

    ax.set_xlabel("M_x(true)", fontsize=10)
    ax.set_xticks(np.linspace(150, 450, 7))
    ax.set_xticklabels([f"{x:.2e}" for x in np.linspace(150, 450, 7)])
    
    ax.set_ylabel("M_x(pred) / M_x(true)", fontsize=10)

    # Add legend
    ax.legend(fontsize=10)

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

        ax.axhline(y=(mean / true_val), color=model_color, linestyle='-', alpha=0.9, label=model_name + " mean")
        ax.axhline(y=(plusOneSigma / true_val), color=model_color, linestyle=':', alpha=0.65, label=model_name + " +1σ")
        ax.axhline(y=(minusOneSigma / true_val), color=model_color, linestyle=':', alpha=0.65, label=model_name + " -1σ")

    # Add a green diagonal line for the perfect prediction ratio = 1
    ax.axhline(y=1.0, color='green', linestyle='-', alpha=1.0)

    # Set labels and title
    ax.set_title("Model Prediction Accuracy Comparison", fontsize=12)

    ax.set_xlabel("M_x(true)", fontsize=10)
    
    ax.set_ylabel("M_x(pred) / M_x(true)", fontsize=10)

    # Add legend
    ax.legend(fontsize=10)

    plt.tight_layout()

    plt.savefig(filename)
    plt.close(fig)
