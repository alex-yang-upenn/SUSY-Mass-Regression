"""
Module Name: preprocess_data

Description:
    Reads the data from ROOT files and stores it into numpy files, with train/test/validation splits. Extracts
    information according to values set in config.py

    The resulting input has shape (N_EVENTS, N_PARTICLES, N_FEATURES), where N_PARTICLES = 4
    and N_FEATURES = 6 for the current dataset. The resulting target has shape (N_EVENTS), with a single target
    mass value for each event.  

    N_FEATURES dimension is [pt, eta, phi, one-hot_1, one-hot_2, one-hot_3], where the one-hot encoded values
    indicate MET, Lepton, or Other particles respectively

Usage:
Author:
Date:
License:
"""
import ROOT
import numpy as np
import os
from sklearn.model_selection import train_test_split

from ROOT_utils import extract_event_features
import config


os.makedirs(config.PROCESSED_DATA_DIRECTORY, exist_ok=True)
os.makedirs(os.path.join(config.PROCESSED_DATA_DIRECTORY, "train"), exist_ok=True)
os.makedirs(os.path.join(config.PROCESSED_DATA_DIRECTORY, "val"), exist_ok=True)
os.makedirs(os.path.join(config.PROCESSED_DATA_DIRECTORY, "test"), exist_ok=True)

np.random.seed(42)


def main():    
    for name in os.listdir(config.RAW_DATA_DIRECTORY):
        # Skip non-ROOT files
        if name[-5:] != ".root":
            continue
        
        print(f"Processing file {name}")       
        file = ROOT.TFile.Open(os.path.join(config.RAW_DATA_DIRECTORY, name))
        
        X, y_mass, y_eta = [], [], []

        test_tree = file.Get("test")
        sample_count = test_tree.GetEntries()
        for entryNum in range(sample_count):
            test_tree.GetEntry(entryNum)
            features = extract_event_features(
                tree=test_tree,
                decay_chain=config.DECAY_CHAIN,
                MET_ids=config.MET_IDS,
                Lepton_ids=config.LEPTON_IDS,
            )
            # Only append if return isn't None
            if features:
                X.append(features)
                y_mass.append(getattr(test_tree, "T1M"))
                y_eta.append(getattr(test_tree, "METEta"))
        
        X = np.array(X, dtype=np.float32)
        y_mass = np.array(y_mass, dtype=np.float32)
        y_eta = np.array(y_eta, dtype=np.float32)

        X_train, X_temp, y_mass_train, y_mass_temp, y_eta_train, y_eta_temp = train_test_split(
            X, y_mass, y_eta, test_size=config.TRAIN_TEST_SPLIT, random_state=42
        )
        X_val, X_test, y_mass_val, y_mass_test, y_eta_val, y_eta_test = train_test_split(
            X_temp, y_mass_temp, y_eta_temp, test_size=0.5, random_state=42
        )

        np_filename = name[0:-5] + ".npz"
        np.savez_compressed(os.path.join(config.PROCESSED_DATA_DIRECTORY, "train", np_filename), X=X_train, y=y_mass_train, y_eta=y_eta_train)
        np.savez_compressed(os.path.join(config.PROCESSED_DATA_DIRECTORY, "val", np_filename ), X=X_val, y=y_mass_val, y_eta=y_eta_val)
        np.savez_compressed(os.path.join(config.PROCESSED_DATA_DIRECTORY, "test", np_filename), X=X_test, y=y_mass_test, y_eta=y_eta_test)
        
        file.Close()
        

if __name__ == "__main__":
    main()