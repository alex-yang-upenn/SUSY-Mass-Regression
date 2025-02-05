from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


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


def cylindrical_to_cartesian(pts, etas, phis):
    """
    Returns
    --------
    tuple: (px, py, pz) from pts, etas, phis
    """
    px = pts * np.cos(phis)
    py = pts * np.sin(phis)
    pz = pts * np.sinh(etas)
    return px, py, pz

def vectorized_lorentz_addition(particles, particle_masses):
    """
    Lorentz vector addition with numpy
    
    Parameters:
    -----------
    particles : np.array, shape=(n_events, n_particles, 6). The first three features are Pt, Eta, Phi.
    particle_masses : np.array, shape=(n_events, n_particles). The mass of each input particle.
    
    Returns:
    --------
    np.array, shape=(n_events). Mass of X for each event
    """

    pts = particles[:, :, 0]
    etas = particles[:, :, 1]
    phis = particles[:, :, 2]

    px, py, pz = cylindrical_to_cartesian(pts, etas, phis)

    E = np.sqrt(px**2 + py**2 + pz**2 + particle_masses**2)

    P_sum = np.stack([
        np.sum(px, axis=1),
        np.sum(py, axis=1),
        np.sum(pz, axis=1),
        np.sum(E, axis=1)
    ], axis=1)

    calc_mass = np.sqrt(np.abs(
        P_sum[:, 3]**2 - 
        (P_sum[:, 0]**2 + P_sum[:, 1]**2 + P_sum[:, 2]**2)
    ))

    return calc_mass

def scale_data(data, scalers, scalable_particle_features):
    scaled_data = data.copy()
    for scaler, idx in zip(scalers, scalable_particle_features):
        values = data[:, :, idx]
        values_flat = values.reshape(-1, 1)

        scaled_values = scaler.transform(values_flat)

        scaled_data[:, :, idx] = scaled_values.reshape(values.shape)
    
    return scaled_data

def calculate_metrics(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    relative_error = np.abs(y_true - y_pred) / np.abs(y_true)
    mean_relative_error = np.mean(relative_error)
    median_relative_error = np.median(relative_error)
    
    return {
        f"{name}_metrics": {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mean_relative_error": float(mean_relative_error),
            "median_relative_error": float(median_relative_error),
        }
    }