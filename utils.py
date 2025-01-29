
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
    n, bins, patches = ax.hist(npData, bins=200, range=(0, max_val), edgecolor="black", alpha=0.5)
    ax.axvline(x=marker, color="red", linestyle="--", linewidth=2)
    ax.axvline(x=mean_val, color="orange", linestyle=":", linewidth=2)   
    ax.axvline(x=mean_val+std_val, color="yellow", linestyle=":", linewidth=2)
    ax.axvline(x=mean_val-std_val, color="yellow", linestyle=":", linewidth=2)    

    # Add labels and format axes
    ax.set_title(title, fontsize=12)

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_xticks(np.linspace(0, max_val, 6))
    ax.set_xticklabels([f'{x:.2e}' for x in np.linspace(0, max_val, 6)])

    ax.set_ylabel("Frequency", fontsize=10)
    
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', label=marker_label),
        Patch(color="blue", label=data_label),
        Line2D([0], [0], color="orange", linestyle=":", label=f"Mean: {mean_val:.2e}"),
        Line2D([0], [0], color="yellow", linestyle=":", label=f"±1σ: {mean_val-std_val:.2e}, {mean_val+std_val:.2e}"),
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

    # Create figure 
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot both histograms with transparency
    n1, bins1, patches1 = ax.hist(npData1, bins=200, range=(0, max_val), 
                                 alpha=0.5, edgecolor='black', color="blue", label=data_label1)
    n2, bins2, patches2 = ax.hist(npData2, bins=200, range=(0, max_val), 
                                 alpha=0.5, edgecolor='black', color="green", label=data_label2)
    ax.axvline(x=marker, color='red', linestyle='--', linewidth=2, label=marker_label)

    # Add labels and format axes
    ax.set_title(title, fontsize=12)

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_xticks(np.linspace(0, max_val, 6))
    ax.set_xticklabels([f"{x:.2e}" for x in np.linspace(0, max_val, 6)])

    ax.set_ylabel("Frequency", fontsize=10)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)

    plt.tight_layout()
    
    # Save to file
    plt.savefig(filename)
    plt.close(fig)