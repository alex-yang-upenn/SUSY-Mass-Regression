"""Utility functions for data loading, preprocessing, and metric calculation.

This module provides functions for coordinate transformations, data scaling and
normalization, dataset loading, Lorentz addition calculations, and model evaluation
metrics for particle physics mass regression.
"""

import os
import pickle

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def cylindrical_to_cartesian(pts, etas, phis):
    """Convert cylindrical coordinates to Cartesian momentum components.

    Args:
        pts: Array of transverse momenta (pT) in GeV/c.
        etas: Array of pseudorapidities (dimensionless).
        phis: Array of azimuthal angles in radians.

    Returns:
        tuple: Three arrays (px, py, pz) representing Cartesian momentum
            components in GeV/c.

    Example:
        >>> px, py, pz = cylindrical_to_cartesian(np.array([50, 60]),
        ...                                        np.array([1.2, 0.5]),
        ...                                        np.array([0.5, 1.0]))
    """
    px = pts * np.cos(phis)
    py = pts * np.sin(phis)
    pz = pts * np.sinh(etas)
    return px, py, pz


def vectorized_lorentz_addition(
    particles, particle_masses, pt_index=0, eta_index=1, phi_index=2
):
    """Sum Lorentz four-vectors across all particles to compute invariant mass.

    Vectorized implementation using NumPy for efficient computation of invariant
    mass from particle kinematics. Converts cylindrical coordinates to Cartesian,
    computes energy, sums four-vectors, and calculates invariant mass.

    Args:
        particles: Array of shape (n_samples, n_particles, n_features) containing
            particle kinematic features.
        particle_masses: Array of shape (n_samples, n_particles) containing the
            mass of each particle in GeV/c^2. Must correspond to particles at the
            same indices.
        pt_index: Int, index along feature dimension for transverse momentum.
            Defaults to 0.
        eta_index: Int, index along feature dimension for pseudorapidity.
            Defaults to 1.
        phi_index: Int, index along feature dimension for azimuthal angle.
            Defaults to 2.

    Returns:
        numpy.ndarray: Array of shape (n_samples,) containing the invariant mass
            in GeV/c^2 for each event, computed from summed four-vectors.

    Example:
        >>> particles = np.random.rand(100, 4, 6)  # 100 events, 4 particles, 6 features
        >>> masses = np.zeros((100, 4))
        >>> inv_mass = vectorized_lorentz_addition(particles, masses)
        >>> print(inv_mass.shape)  # (100,)
    """
    pts = particles[:, :, 0]
    etas = particles[:, :, 1]
    phis = particles[:, :, 2]

    px, py, pz = cylindrical_to_cartesian(pts, etas, phis)

    E = np.sqrt(px**2 + py**2 + pz**2 + particle_masses**2)

    P_sum = np.stack(
        [np.sum(px, axis=1), np.sum(py, axis=1), np.sum(pz, axis=1), np.sum(E, axis=1)],
        axis=1,
    )

    calc_mass = np.sqrt(
        np.abs(
            P_sum[:, 3] ** 2 - (P_sum[:, 0] ** 2 + P_sum[:, 1] ** 2 + P_sum[:, 2] ** 2)
        )
    )

    return calc_mass


def normalize_data(train, scalable_particle_features):
    """Normalize specified features using StandardScaler.

    Fits sklearn StandardScaler on training data for specified features and
    transforms them. Each feature is scaled independently across all particles
    and samples to have zero mean and unit variance.

    Args:
        train: Array of shape (n_samples, n_particles, n_features) containing
            training data.
        scalable_particle_features: List of int, indices of features to normalize.
            For example, [0, 1, 2] normalizes the first 3 features (pt, eta, phi).

    Returns:
        tuple: Two-element tuple containing:
            - numpy.ndarray: Normalized training dataset with same shape as input.
            - list: List of sklearn.preprocessing.StandardScaler objects, one per
              normalized feature, in the same order as scalable_particle_features.

    Example:
        >>> X_train = np.random.rand(1000, 4, 6)
        >>> X_normalized, scalers = normalize_data(X_train, [0, 1, 2])
        >>> # scalers[0] is the scaler for feature 0, etc.
    """
    scalers = []
    scaled_train = train.copy()
    for i in scalable_particle_features:
        values = train[:, :, i]
        values_flat = values.reshape(-1, 1)

        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(values_flat)

        scaled_train[:, :, i] = scaled_values.reshape(values.shape)
        scalers.append(scaler)

    return scaled_train, scalers


def scale_data(data, scalers, scalable_particle_features):
    """Transform data using pre-fitted StandardScalers.

    Applies pre-fitted scalers to transform specified features in the dataset.
    Used to scale validation/test data using statistics from training data.

    Args:
        data: Array of shape (n_samples, n_particles, n_features) to be scaled.
        scalers: List of sklearn.preprocessing.StandardScaler objects, pre-fitted
            on training data.
        scalable_particle_features: List of int, feature indices corresponding to
            each scaler. scalers[i] is applied to data[:, :, scalable_particle_features[i]].

    Returns:
        numpy.ndarray: Scaled dataset with same shape as input.

    Example:
        >>> X_test = np.random.rand(200, 4, 6)
        >>> X_test_scaled = scale_data(X_test, scalers, [0, 1, 2])
    """
    scaled_data = data.copy()
    for scaler, idx in zip(scalers, scalable_particle_features):
        values = data[:, :, idx]
        values_flat = values.reshape(-1, 1)

        scaled_values = scaler.transform(values_flat)

        scaled_data[:, :, idx] = scaled_values.reshape(values.shape)

    return scaled_data


def load_data(data_directory):
    """Load and preprocess data for model training.

    Loads all .npz files from train/val/test subdirectories, concatenates them,
    applies pre-fitted scalers from preprocessing, and transposes to model input
    format (n_samples, n_features, n_particles).

    Args:
        data_directory: Str, path to directory containing train/val/test subdirectories
            and scaler pickle files (x_scaler_0.pkl, x_scaler_1.pkl, x_scaler_2.pkl,
            y_scaler.pkl).

    Returns:
        tuple: Six arrays in order:
            - X_train_scaled: Scaled training features, shape (n_train, n_features, n_particles).
            - y_train_scaled: Scaled training targets, shape (n_train, 1).
            - X_val_scaled: Scaled validation features, shape (n_val, n_features, n_particles).
            - y_val_scaled: Scaled validation targets, shape (n_val, 1).
            - X_test_scaled: Scaled test features, shape (n_test, n_features, n_particles).
            - y_test_scaled: Scaled test targets, shape (n_test, 1).

    Note:
        This function deletes intermediate variables to save memory after loading.

    Example:
        >>> X_train, y_train, X_val, y_val, X_test, y_test = load_data("processed_data")
    """
    X_trains, y_trains = [], []
    X_vals, y_vals = [], []
    X_tests, y_tests = [], []

    files = [
        f
        for f in os.listdir(os.path.join(data_directory, "test"))
        if f.endswith(".npz")
    ]

    for name in tqdm(files, desc="Loading data"):
        train = np.load(os.path.join(data_directory, "train", name))
        val = np.load(os.path.join(data_directory, "val", name))
        test = np.load(os.path.join(data_directory, "test", name))

        X_trains.append(train["X"])
        y_trains.append(train["y"])
        X_vals.append(val["X"])
        y_vals.append(val["y"])
        X_tests.append(test["X"])
        y_tests.append(test["y"])

    X_train = np.concatenate(X_trains, axis=0)
    y_train = np.concatenate(y_trains, axis=0)
    X_val = np.concatenate(X_vals, axis=0)
    y_val = np.concatenate(y_vals, axis=0)
    X_test = np.concatenate(X_tests, axis=0)
    y_test = np.concatenate(y_tests, axis=0)

    scalers = []
    for i in range(3):
        with open(os.path.join(data_directory, f"x_scaler_{i}.pkl"), "rb") as f:
            scalers.append(pickle.load(f))

    X_train_scaled = scale_data(X_train, scalers, [0, 1, 2])
    X_val_scaled = scale_data(X_val, scalers, [0, 1, 2])
    X_test_scaled = scale_data(X_test, scalers, [0, 1, 2])

    with open(os.path.join(data_directory, "y_scaler.pkl"), "rb") as f:
        y_scaler = pickle.load(f)
    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1))
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

    del X_train, X_val, X_test, y_train, y_val, y_test
    del X_trains, X_vals, X_tests, y_trains, y_vals, y_tests
    del scalers, y_scaler

    # Change from (batchsize, num particles, num features) to (batchsize, num features, num particles)
    X_train_scaled = X_train_scaled.transpose(0, 2, 1)
    X_val_scaled = X_val_scaled.transpose(0, 2, 1)
    X_test_scaled = X_test_scaled.transpose(0, 2, 1)

    return (
        X_train_scaled,
        y_train_scaled,
        X_val_scaled,
        y_val_scaled,
        X_test_scaled,
        y_test_scaled,
    )


def load_data_original(data_directory):
    """Load raw unscaled data without preprocessing transformations.

    Loads all .npz files from train/val/test subdirectories and concatenates them.
    Unlike load_data(), this function does NOT apply scaling or transposition,
    returning data in original (n_samples, n_particles, n_features) format.

    Args:
        data_directory: Str, path to directory containing train/val/test subdirectories
            with .npz files.

    Returns:
        tuple: Six arrays in order:
            - X_train: Unscaled training features, shape (n_train, n_particles, n_features).
            - y_train: Unscaled training targets, shape (n_train,).
            - X_val: Unscaled validation features, shape (n_val, n_particles, n_features).
            - y_val: Unscaled validation targets, shape (n_val,).
            - X_test: Unscaled test features, shape (n_test, n_particles, n_features).
            - y_test: Unscaled test targets, shape (n_test,).

    Note:
        Used for Lorentz addition baseline and transformed data evaluation where
        original coordinate values are needed.

    Example:
        >>> X_train, y_train, X_val, y_val, X_test, y_test = load_data_original("processed_data")
    """
    X_trains, y_trains = [], []
    X_vals, y_vals = [], []
    X_tests, y_tests = [], []

    files = [
        f
        for f in os.listdir(os.path.join(data_directory, "test"))
        if f.endswith(".npz")
    ]

    for name in tqdm(files, desc="Loading data"):
        train = np.load(os.path.join(data_directory, "train", name))
        val = np.load(os.path.join(data_directory, "val", name))
        test = np.load(os.path.join(data_directory, "test", name))

        X_trains.append(train["X"])
        y_trains.append(train["y"])
        X_vals.append(val["X"])
        y_vals.append(val["y"])
        X_tests.append(test["X"])
        y_tests.append(test["y"])

    X_train = np.concatenate(X_trains, axis=0)
    y_train = np.concatenate(y_trains, axis=0)
    X_val = np.concatenate(X_vals, axis=0)
    y_val = np.concatenate(y_vals, axis=0)
    X_test = np.concatenate(X_tests, axis=0)
    y_test = np.concatenate(y_tests, axis=0)

    del X_trains, X_vals, X_tests, y_trains, y_vals, y_tests

    return X_train, y_train, X_val, y_val, X_test, y_test


def calculate_metrics(y_true, y_pred, name):
    """Calculate regression metrics for model evaluation.

    Computes standard regression metrics including MSE, RMSE, MAE, R^2, and
    relative errors to evaluate model performance.

    Args:
        y_true: Array of true target values (ground truth masses).
        y_pred: Array of predicted target values (predicted masses).
        name: Str, identifier for the model (e.g., "gnn_baseline") used as
            dictionary key prefix.

    Returns:
        dict: Nested dictionary with structure {f"{name}_metrics": {...}} containing:
            - mse: Mean squared error.
            - rmse: Root mean squared error.
            - mae: Mean absolute error.
            - r2: R-squared (coefficient of determination).
            - mean_relative_error: Mean of |y_true - y_pred| / |y_true|.
            - median_relative_error: Median of |y_true - y_pred| / |y_true|.

    Example:
        >>> metrics = calculate_metrics(y_true, y_pred, "gnn_baseline")
        >>> print(metrics["gnn_baseline_metrics"]["rmse"])
    """
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
