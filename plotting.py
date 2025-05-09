import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np


def create_1var_histogram_with_marker(data, data_label, marker, marker_label, title, x_label, filename):
    """
    Creates a histogram from one dataset with a vertical marker line and saves it to a file. Uses 200 bins
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
    Creates a histogram comparing two datasets with a vertical marker line and saves it to a file. Uses 200 bins.
    Histograms are semi-transparent (alpha=0.5) to show overlap. First dataset is blue, second is green
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
    plt.style.use("default")
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    colors = cm.tab10(np.linspace(0, 1, len(model_performance_dict)))

    # Plot each model
    for i, (model_name, model_data) in enumerate(model_performance_dict.items()):
        median = model_data["median"]
        plusOneSigma = model_data["+1σ"]
        minusOneSigma = model_data["-1σ"]

        model_color = colors[i]

        ax.scatter(
            median["M_x(true)"], 
            median["ratio"],
            color=model_color,
            alpha=0.9,
            label=model_name + " median",
        )

        ax.scatter(
            plusOneSigma["M_x(true)"],
            plusOneSigma["ratio"],
            color=model_color,
            alpha=0.65,
            label=model_name + " +1σ",
        )

        ax.scatter(
            minusOneSigma["M_x(true)"],
            minusOneSigma["ratio"],
            color=model_color,
            alpha=0.65,
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
    plt.style.use("default")
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    colors = cm.tab10(np.linspace(0, 1, len(model_performance_dict)))

    # Plot each model
    for i, (model_name, model_data) in enumerate(model_performance_dict.items()):
        mean = model_data["mean"]
        plusOneSigma = model_data["+1σ"]
        minusOneSigma = model_data["-1σ"]

        model_color = colors[i]

        ax.axhline(y=mean, color=model_color, linestyle='-', alpha=0.9, label=model_name + " mean")
        ax.axhline(y=plusOneSigma, color=model_color, linestyle=':', alpha=0.65, label=model_name + " +1σ")
        ax.axhline(y=minusOneSigma, color=model_color, linestyle=':', alpha=0.65, label=model_name + " -1σ")

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
