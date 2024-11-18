"""
Module Name: Preprocess

Description:
    This module handles reading the data from ROOT files, and storing it into numpy files
    It handles the train/test split, and chunks the resulting numpy files to improve memory usage.

    The ROOT files currently being used have decay structure X -> qqLv, with the neutrino as the sole 
    contributor to MET. Thus, the features will be organized as follows:
    
    Sample Input (Length 14 float vector)
    { [1]q1 Pt, [2]q1 Eta, [3]q1 Phi, [4]q1 Mass, [5]q2 Pt, ... , [9]lepton Pt, ... , [13]MET Pt, [14]MET Phi }
    Sample Output (Float)
    X Mass

Usage:
Author:
Date:
License:
"""

import ROOT
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os
import pickle

import utils


# Parameters
DATA_DIRECTORY = "MassRegressionData"
PROCESSED_DATA_DIRECTORY = "Preprocessed_Data"
TRAIN_TEST_SPLIT = 0.2
TARGET_BRANCHES = ["P1"]
UNDETECTED_PARTICLES = [12]
DNN_NUM_FEATURES = 14

os.makedirs(PROCESSED_DATA_DIRECTORY, exist_ok=True)

np.random.seed(42)


def main():
    idx = 0
    inputs = []
    outputs = []
    outputs_eta = []
    for name in os.listdir(DATA_DIRECTORY):
        if name[-5:] != ".root":
            continue

        print(f"Processing file {name}")
        filename = os.path.join(DATA_DIRECTORY, name)
        file = ROOT.TFile.Open(filename)

        info_tree = file.Get("info")
        info_tree.GetEntry(0)
        truthM = list(getattr(info_tree, "truthM"))
        truthID = list(getattr(info_tree, "truthId"))

        test_tree = file.Get("test")
        sample_count = test_tree.GetEntries()

        for entryNum in range(sample_count):
            test_tree.GetEntry(entryNum)
            
            features = utils.single_entry_extract_features(
                tree=test_tree,
                branches=TARGET_BRANCHES,
                truthM=truthM,
                truthID=truthID,
                MET_ids=UNDETECTED_PARTICLES,
                clip_eta=True
            )

            if (len(features) > 0):
                inputs.append(features + [getattr(test_tree, "METPt"), getattr(test_tree, "METPhi")])
                outputs.append(getattr(test_tree, "T1M"))
                outputs_eta.append(getattr(test_tree, "METEta"))
                idx += 1
        
        file.Close()
        print(f"{idx + 1} samples added so far")
    
    # Convert
    inputs = np.array(inputs, dtype=np.float32)
    outputs = np.array(outputs, dtype=np.float32)
    outputs_eta = np.array(outputs_eta, dtype=np.float32)

    # Shuffle
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    inputs, outputs, outputs_eta = inputs[indices], outputs[indices], outputs_eta[indices]

    # Scale inputs
    X_train, X_temp, y_train, y_temp, y_eta_train, y_eta_temp = train_test_split(
        inputs, outputs, outputs_eta, test_size=0.2, random_state=42
    )
    X_val, X_test, y_val, y_test, y_eta_val, y_eta_test = train_test_split(
        X_temp, y_temp, y_eta_temp, test_size=0.5, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    np.savez_compressed(os.path.join(PROCESSED_DATA_DIRECTORY, "train.npz"), X=X_train_scaled, y=y_train, y_eta=y_eta_train)
    np.savez_compressed(os.path.join(PROCESSED_DATA_DIRECTORY, "val.npz"), X=X_val_scaled, y=y_val, y_eta=y_eta_val)
    np.savez_compressed(os.path.join(PROCESSED_DATA_DIRECTORY, "test.npz"), X=X_test_scaled, y=y_test, y_eta=y_eta_test)
    with open(os.path.join(PROCESSED_DATA_DIRECTORY, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == "__main__":
    main()