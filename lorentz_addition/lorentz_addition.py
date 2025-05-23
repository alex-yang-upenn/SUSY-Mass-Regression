"""
Module Name: lorentz_addition

Description:
    This module handles inspecting the preprocessed numpy files, to verify that the data is correct.
    It also attempts to compute unknwon particle mass with a naive lorentz addition of the decay chain
    products (set MET_eta = 0)

Usage:
Author:
Date:
License:
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(ROOT_DIR)

import ROOT
import numpy as np
import os

import config
from plotting import *
from utils import *


def main():
    os.makedirs(os.path.join(SCRIPT_DIR, "graphs"), exist_ok=True)

    single_file_metrics = {}

    for name in config.EVAL_DATA_FILES:
        print(f"Processing file {name}")

        # Load in data
        train = np.load(os.path.join(config.PROCESSED_DATA_DIRECTORY, "train", name))
        val = np.load(os.path.join(config.PROCESSED_DATA_DIRECTORY, "val", name))
        test = np.load(os.path.join(config.PROCESSED_DATA_DIRECTORY, "test", name))
        
        X_train, y_train, y_eta_train = train["X"], train["y"], train["y_eta"]
        X_val, y_val, y_eta_val = val["X"], val["y"], val["y_eta"]
        X_test, y_test, y_eta_test = test["X"], test["y"], test["y_eta"]

        X_all = np.concatenate([X_train, X_val, X_test])
        y_eta_all = np.concatenate([y_eta_train, y_eta_val, y_eta_test])
        y_all = np.concatenate([y_train, y_val, y_test])
        
        print(f"Found {len(y_all)} events")

        # Obtain true mass from ROOT file
        root_name = name.replace(".npz", ".root")
        root_file = ROOT.TFile.Open(os.path.join(config.RAW_DATA_DIRECTORY, root_name))
        test_tree = root_file.Get("test")
        test_tree.GetEntry(0)
        y_true = getattr(test_tree, "T1M")
        root_file.Close()
        
        # Sanity check on our "True" values. Both should equal y_true, obtained from ROOT file
        # y_all: train/test/val targets
        # y_true_calc: Lorentz Addition with full information
        particle_masses = np.zeros((X_all.shape[0], X_all.shape[1]))

        full_decay_products = X_all.copy()
        full_decay_products[:, 3, 1] = y_eta_all
        y_true_calc = vectorized_lorentz_addition(full_decay_products, particle_masses)

        # Error check
        target_y_errors = np.sum(~np.isclose(y_true, y_all, rtol=1e-5))
        calc_y_errors = np.sum(~np.isclose(y_true, y_true_calc, rtol=1e-5))
        print(f"{target_y_errors} samples had significant errors in target mass value")
        print(f"{calc_y_errors} samples had significant errors in calculated mass value")

        # Lorentz Addition with available information (no MET eta). Baseline for model evaluation
        y_lorentz = vectorized_lorentz_addition(X_all, particle_masses)

        lorentz_metrics = calculate_metrics(y_all, y_lorentz, "lorentz")
        single_file_metrics[name] = lorentz_metrics

        # Plot for lorentz addition summation
        create_1var_histogram_with_marker(
            data=y_lorentz,
            data_label="Lorentz Addition Prediction",
            marker=y_true,
            marker_label="True Mass",
            title=f"Mass Regression for {name}",
            x_label="Mass (GeV / c^2)",
            filename=os.path.join(SCRIPT_DIR, f"graphs/{name}.png"),
        )

if __name__ == "__main__":
    main()