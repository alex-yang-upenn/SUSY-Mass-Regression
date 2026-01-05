"""Preprocessing script for SUSY signal event data.

This module converts raw ROOT files containing SUSY signal events into preprocessed
NumPy arrays with train/validation/test splits for mass regression tasks.

The script extracts particle features (pt, eta, phi) and applies one-hot encoding
for particle types (MET, Lepton, Gluon, Other). Output shape is (N_EVENTS, N_PARTICLES,
N_FEATURES) for inputs and (N_EVENTS,) for target masses.

Typical usage example:
    python preprocess_data.py              # uses config.yaml
    python preprocess_data.py --dataset set2  # uses config_set2.yaml
"""

import os

import numpy as np
import ROOT
from sklearn.model_selection import train_test_split

from config_loader import get_dataset_type_from_args, load_config
from ROOT_utils import extract_event_features

np.random.seed(42)


def main():
    """Process raw ROOT signal data files into NumPy format with train/val/test splits.

    Iterates through all ROOT files in the configured raw data directory, extracts
    particle features and target masses (T1M) from decay chains, and saves split
    datasets as compressed NumPy arrays. Also extracts METEta as auxiliary target.

    The function:
        1. Loads configuration (dataset 1 or set2) from command line args
        2. Creates output directories for train/val/test splits
        3. Processes each ROOT file to extract event features
        4. Performs 80/10/10 train/val/test split
        5. Saves compressed .npz files with X (features), y (mass), and y_eta
    """
    # Load configuration based on command line argument
    dataset_type = get_dataset_type_from_args()
    config = load_config(dataset_type)

    os.makedirs(config["PROCESSED_DATA_DIRECTORY"], exist_ok=True)
    os.makedirs(
        os.path.join(config["PROCESSED_DATA_DIRECTORY"], "train"), exist_ok=True
    )
    os.makedirs(os.path.join(config["PROCESSED_DATA_DIRECTORY"], "val"), exist_ok=True)
    os.makedirs(os.path.join(config["PROCESSED_DATA_DIRECTORY"], "test"), exist_ok=True)
    for name in os.listdir(config["RAW_DATA_DIRECTORY"]):
        # Skip non-ROOT files
        if name[-5:] != ".root":
            continue

        print(f"Processing file {name}")
        file = ROOT.TFile.Open(os.path.join(config["RAW_DATA_DIRECTORY"], name))

        X, y_mass, y_eta = [], [], []

        test_tree = file.Get("test")
        sample_count = test_tree.GetEntries()
        for entryNum in range(sample_count):
            test_tree.GetEntry(entryNum)
            # Build kwargs for extract_event_features, including optional Gluon_ids for set2
            kwargs = {
                "tree": test_tree,
                "decay_chains": config["DECAY_CHAIN"],
                "MET_ids": config["MET_IDS"],
                "Lepton_ids": config["LEPTON_IDS"],
            }
            if "GLUON_IDS" in config:
                kwargs["Gluon_ids"] = config["GLUON_IDS"]

            features = extract_event_features(**kwargs)
            # Only append if return isn't None
            if features:
                X.append(features)
                y_mass.append(getattr(test_tree, "T1M"))
                y_eta.append(getattr(test_tree, "METEta"))

        X = np.array(X, dtype=np.float32)
        y_mass = np.array(y_mass, dtype=np.float32)
        y_eta = np.array(y_eta, dtype=np.float32)

        X_train, X_temp, y_mass_train, y_mass_temp, y_eta_train, y_eta_temp = (
            train_test_split(
                X, y_mass, y_eta, test_size=config["TRAIN_TEST_SPLIT"], random_state=42
            )
        )
        X_val, X_test, y_mass_val, y_mass_test, y_eta_val, y_eta_test = (
            train_test_split(
                X_temp, y_mass_temp, y_eta_temp, test_size=0.5, random_state=42
            )
        )

        np_filename = name[0:-5] + ".npz"
        np.savez_compressed(
            os.path.join(config["PROCESSED_DATA_DIRECTORY"], "train", np_filename),
            X=X_train,
            y=y_mass_train,
            y_eta=y_eta_train,
        )
        np.savez_compressed(
            os.path.join(config["PROCESSED_DATA_DIRECTORY"], "val", np_filename),
            X=X_val,
            y=y_mass_val,
            y_eta=y_eta_val,
        )
        np.savez_compressed(
            os.path.join(config["PROCESSED_DATA_DIRECTORY"], "test", np_filename),
            X=X_test,
            y=y_mass_test,
            y_eta=y_eta_test,
        )

        file.Close()


if __name__ == "__main__":
    main()
