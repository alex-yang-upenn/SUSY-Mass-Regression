"""
Module Name: preprocess_data

Description:
    Reads the data from ROOT files and stores it into numpy files, with train/test/validation splits.

    The data dimension is (n, p, 6), where n is the number of events, p is the number of particles
    at the end of the decay chain for each event, and there are 6 features per particle:

    [pt, eta, phi, one-hot_1, one-hot_2, one-hot_3]
    
    where the one-hot encoded values indicate MET, Lepton, or other particles respectively

Usage:
Author:
Date:
License:
"""

import ROOT
import numpy as np
from sklearn.model_selection import train_test_split

import os


# Parameters
DATA_DIRECTORY = "raw_data"
PROCESSED_DATA_DIRECTORY = "processed_data"
DECAY_CHAIN = "P1"
MET_IDS = [12]
LEPTON_IDS = [11]
TRAIN_TEST_SPLIT = 0.2

os.makedirs(PROCESSED_DATA_DIRECTORY, exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIRECTORY, "train"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIRECTORY, "val"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIRECTORY, "test"), exist_ok=True)

np.random.seed(42)


def extract_event_features(tree, decay_chain, MET_ids, Lepton_ids):
    """
    Extracts custom event features. Filters out events with any |eta| > 3.4.
    
    Parameters:
    -----------
    tree : ROOT.TTree
    decay_chain : str
        The prefix for the decay chain to extract features from. This string will be concatenated with the
        suffixes "Id", "Pt", "Eta", and "Phi" to access the corresponding branches
    MET_ids : set
        Set of IDs that indicate MET particles
    Lepton_ids: set
        Set of IDs that indicate Lepton particles
    other_ids: set
        Set of IDs that indicate other particles
    
    Returns:
    --------
    list
        Features for a single event, as described at the top of the file, with dimensions (p, 6)
    """
    features = []

    Id = list(getattr(tree, decay_chain + "Id"))
    Pt = list(getattr(tree, decay_chain + "Pt"))
    Eta = list(getattr(tree, decay_chain + "Eta"))
    Phi = list(getattr(tree, decay_chain + "Phi"))
    
    for j in range(len(Id)):
        if abs(Eta[j]) > 3.4:
            return None  # Drops entire event
        
        if Id[j] in MET_ids:
            one_hot = [1, 0, 0]
            Eta[j] = 0
        elif Id[j] in Lepton_ids:
            one_hot = [0, 1, 0]
        else:
            one_hot = [0, 0, 1]
        
        features.append([Pt[j], Eta[j], Phi[j]] + one_hot)
    
    return features

def main():    
    for name in os.listdir(DATA_DIRECTORY):
        if name[-5:] != ".root":
            continue
        
        print(f"Processing file {name}")       
        file = ROOT.TFile.Open(os.path.join(DATA_DIRECTORY, name))
        
        X = []
        y_mass = []
        y_eta = []

        test_tree = file.Get("test")
        sample_count = test_tree.GetEntries()
        for entryNum in range(sample_count):
            test_tree.GetEntry(entryNum)
            features = extract_event_features(
                tree=test_tree,
                decay_chain=DECAY_CHAIN,
                MET_ids=MET_IDS,
                Lepton_ids=LEPTON_IDS,
            )
            if features:
                X.append(features)
                y_mass.append(getattr(test_tree, "T1M"))
                y_eta.append(getattr(test_tree, "METEta"))
        
        X = np.array(X, dtype=np.float32)
        y_mass = np.array(y_mass, dtype=np.float32)
        y_eta = np.array(y_eta, dtype=np.float32)

        X_train, X_temp, y_mass_train, y_mass_temp, y_eta_train, y_eta_temp = train_test_split(
            X, y_mass, y_eta, test_size=TRAIN_TEST_SPLIT, random_state=42
        )
        X_val, X_test, y_mass_val, y_mass_test, y_eta_val, y_eta_test = train_test_split(
            X_temp, y_mass_temp, y_eta_temp, test_size=0.5, random_state=42
        )

        np_filename = name[0:-5] + ".npz"
        np.savez_compressed(os.path.join(PROCESSED_DATA_DIRECTORY, "train", np_filename), X=X_train, y=y_mass_train, y_eta=y_eta_train)
        np.savez_compressed(os.path.join(PROCESSED_DATA_DIRECTORY, "val", np_filename ), X=X_val, y=y_mass_val, y_eta=y_eta_val)
        np.savez_compressed(os.path.join(PROCESSED_DATA_DIRECTORY, "test", np_filename), X=X_test, y=y_mass_test, y_eta=y_eta_test)
        
        file.Close()
        

if __name__ == "__main__":
    main()