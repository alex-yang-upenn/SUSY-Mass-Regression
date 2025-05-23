import numpy as np
import os
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


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


def vectorized_lorentz_addition(particles, particle_masses, pt_index=0, eta_index=1, phi_index=2):
    """
    Sums the lorentz vectors of all particles, for each event. Uses numpy functions for efficiency.
    
    Args:
        particles (numpy.ndarray):
            Array should be of shape (n_samples, n_particles, n_features)
        particle_masses (numpy.ndarray): 
            Array should be of shape (n_events, n_particles) each entry contains the corresponding mass
            of the particle at the same index in `particles`
        pt_index (int): Index along the third dimension of the input that corresponds to transverse momentum
        eta_index (int): Index along the third dimension of the input that corresponds to pseudorapidity
        phi_index (int): Index along the third dimension of the input that corresponds to azimuthal angle
    
    Returns:
        numpy.ndarray:
            Array with shape (n_events,). Each value is the mass of the unknown particle X for the
            corresponding event
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


def normalize_data(train, scalable_particle_features):
    """
    Normalizes specified model inputs across the training dataset, using sklearn's Standard Scaler.

    Args:
        train (numpy.ndarray): 
            Model inputs. Should be a numpy array with shape (n_samples, n_particles, n_features)
        scalable_particle_features (list of int):
            List of indices specifiying which features to normalize. E.x. scalable_particle_features=[0, 1]
            normalizes the first 2 features, across all particles across all samples.
    
    Returns:
        (numpy.ndarray, sklearn.preprocessing.StandardScaler): 
            First Entry: The training dataset, after normalization.
            
            Second Entry: StandardScalers corresponding to each normalized feature.
            E.x. scalable_particle_features=[0, 1] returns a length 2 list, index 0 contains
            the scaler used for the first feature, etc.
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
    """
    Scales a selection of features in a (None, num_particles, num_features) dimension dataset

    Args:
        data (numpy.ndarray):
            np.array with shape (None, num_particles, num_features)
        scalers (List of StandardScalers):
            List of scalers to apply to the data
        scalable_particle_features (List of int):
            The corresponding indices at which to apply each of the scalers. E.x. scalers[0] will be applied
            to data[:, :, scalable_particle_features[0]]
    """
    scaled_data = data.copy()
    for scaler, idx in zip(scalers, scalable_particle_features):
        values = data[:, :, idx]
        values_flat = values.reshape(-1, 1)

        scaled_values = scaler.transform(values_flat)

        scaled_data[:, :, idx] = scaled_values.reshape(values.shape)
    
    return scaled_data


def load_data(data_directory):
    """
    Load data from files, normalize, and prepare TF datasets
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    X_trains, y_trains = [], []
    X_vals, y_vals = [], []
    X_tests, y_tests = [], []

    files = [f for f in os.listdir(os.path.join(data_directory, "test")) if f.endswith(".npz")]
    
    for name in tqdm(files, desc="Loading data"):
        train = np.load(os.path.join(data_directory, "train", name))
        val = np.load(os.path.join(data_directory, "val", name))
        test = np.load(os.path.join(data_directory, "test", name))

        X_trains.append(train['X'])
        y_trains.append(train['y'])
        X_vals.append(val['X'])
        y_vals.append(val['y'])
        X_tests.append(test['X'])
        y_tests.append(test['y'])

    X_train = np.concatenate(X_trains, axis=0)
    y_train = np.concatenate(y_trains, axis=0)
    X_val = np.concatenate(X_vals, axis=0)
    y_val = np.concatenate(y_vals, axis=0)
    X_test = np.concatenate(X_tests, axis=0)
    y_test = np.concatenate(y_tests, axis=0)
    
    scalers = []
    for i in range(3):
        with open(os.path.join(data_directory, f"x_scaler_{i}.pkl"), 'rb') as f:
            scalers.append(pickle.load(f))

    X_train_scaled = scale_data(X_train, scalers, [0, 1, 2])
    X_val_scaled = scale_data(X_val, scalers, [0, 1, 2])
    X_test_scaled = scale_data(X_test, scalers, [0, 1, 2])
    
    with open(os.path.join(data_directory, "y_scaler.pkl"), 'rb') as f:
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

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled


def load_data_original(data_directory):
    """
    Load data from files but does not scale or transpose
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    X_trains, y_trains = [], []
    X_vals, y_vals = [], []
    X_tests, y_tests = [], []

    files = [f for f in os.listdir(os.path.join(data_directory, "test")) if f.endswith(".npz")]
    
    for name in tqdm(files, desc="Loading data"):
        train = np.load(os.path.join(data_directory, "train", name))
        val = np.load(os.path.join(data_directory, "val", name))
        test = np.load(os.path.join(data_directory, "test", name))

        X_trains.append(train['X'])
        y_trains.append(train['y'])
        X_vals.append(val['X'])
        y_vals.append(val['y'])
        X_tests.append(test['X'])
        y_tests.append(test['y'])

    X_train = np.concatenate(X_trains, axis=0)
    y_train = np.concatenate(y_trains, axis=0)
    X_val = np.concatenate(X_vals, axis=0)
    y_val = np.concatenate(y_vals, axis=0)
    X_test = np.concatenate(X_tests, axis=0)
    y_test = np.concatenate(y_tests, axis=0)

    del X_trains, X_vals, X_tests, y_trains, y_vals, y_tests

    return X_train, y_train, X_val, y_val, X_test, y_test


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