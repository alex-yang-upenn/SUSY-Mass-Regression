"""
Module Name: Inspect_Data

Description:
    This module handles inspecting the preprocessed numpy files, to verify that the data is evenly distributed
    and inputs and outputs are ordered correctly, relative to each other.

Usage:
Author:
Date:
License:
"""

import ROOT
import math
import matplotlib.pyplot as plt
import numpy as np

import os
import pickle

import utils


# Parameters
DATA_DIRECTORY = "Preprocessed_Data"


def InspectData(X, y, y_eta, filename, plotTitle):
    # Unscale data
    with open('Preprocessed_Data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    X = scaler.inverse_transform(X)

    # Frequency Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(y,
             bins=np.arange(195, 405, 10),
             edgecolor="black",
             alpha=0.7)
    plt.title(plotTitle)
    plt.xlabel("Mass (GeV / c^2)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIRECTORY, filename))
    plt.close()

     # Spot check
    indices = np.random.choice(len(y), size=500, replace=False)
    noticeableErrors = 0
    for i in indices:
        q_1 = utils.create_lorentz_vector(X[i][0], X[i][1], X[i][2], X[i][3])
        q_2 = utils.create_lorentz_vector(X[i][4], X[i][5], X[i][6], X[i][7])
        Lept = utils.create_lorentz_vector(X[i][8], X[i][9], X[i][10], X[i][11])
        v = utils.create_lorentz_vector(X[i][12], y_eta[i], X[i][13], 0)
        X_particle = q_1 + q_2 + Lept + v
        mass = X_particle.M()
        if not math.isclose(mass, y[i], rel_tol=1e-5):
            noticeableErrors += 1
            print(f"Inaccuracy at sample {i}. Expected mass of {y[i]}, calculated mass of {mass}")
    print(f"{noticeableErrors} out of {len(indices)} samples with noticeable errors")
    del indices

def main():
    print("Inspecting training data")
    train = np.load(os.path.join(DATA_DIRECTORY, "train.npz"))
    X_train, y_train, y_eta_train = train['X'], train['y'], train['y_eta']
    InspectData(
        X=X_train,
        y=y_train,
        y_eta=y_eta_train,
        filename="y_train_distribution.png",
        plotTitle="X Particle Mass Distribution - Training Data"
    )
    del train, X_train, y_train, y_eta_train

    print("Inspecting validation data")
    val = np.load(os.path.join(DATA_DIRECTORY, "val.npz"))
    X_val, y_val, y_eta_val = val['X'], val['y'], val['y_eta']
    InspectData(
        X=X_val,
        y=y_val,
        y_eta=y_eta_val,
        filename="y_val_distribution.png",
        plotTitle="X Particle Mass Distribution - Validation Data"
    )
    del val, X_val, y_val, y_eta_val

    print("Inspecting testing data")
    test = np.load(os.path.join(DATA_DIRECTORY, "test.npz"))
    X_test, y_test, y_eta_test = test['X'], test['y'], test['y_eta']
    InspectData(
        X=X_test,
        y=y_test,
        y_eta=y_eta_test,
        filename="y_test_distribution.png",
        plotTitle="X Particle Mass Distribution - Test Data"
    )
    del test, X_test, y_test, y_eta_test


if __name__ == "__main__":
    main()
