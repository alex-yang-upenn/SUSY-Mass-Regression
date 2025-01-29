"""
Module Name: lorentz_addition

Description:
    This module handles inspecting the preprocessed numpy files, to verify that the data is correct.
    It also attempts a naive lorentz addition of the decay chain products, with with MET_eta = 0

Usage:
Author:
Date:
License:
"""
import ROOT
import numpy as np
import os
import random
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils import create_1var_histogram_with_marker

DATA_DIRECTORY = os.path.join(os.path.dirname(SCRIPT_DIR), "raw_data")
PROCESSED_DATA_DIRECTORY = os.path.join(os.path.dirname(SCRIPT_DIR), "processed_data")
OUTPUT_IMAGE_DIRECTORY = os.path.join(SCRIPT_DIR, "graphs")

os.makedirs(OUTPUT_IMAGE_DIRECTORY, exist_ok=True)

random.seed(42)


def create_lorentz_vector(pt, eta, phi, mass):
    """
    Returns:
    --------
    ROOT.TLorentzVector: Lorentz vector from pt, eta, phi, mass
    """
    vec = ROOT.TLorentzVector()
    vec.SetPtEtaPhiM(pt, eta, phi, mass)
    return vec

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

def main():
    sampleFiles = random.sample(os.listdir(DATA_DIRECTORY), 5)

    for name in sampleFiles:
        print(f"Processing file {name}")

        # Load in data
        np_name = name.replace(".root", ".npz")
        train = np.load(os.path.join(PROCESSED_DATA_DIRECTORY, "train", np_name))
        val = np.load(os.path.join(PROCESSED_DATA_DIRECTORY, "val", np_name))
        test = np.load(os.path.join(PROCESSED_DATA_DIRECTORY, "test", np_name))
        
        X_train, y_train, y_eta_train = train["X"], train["y"], train["y_eta"]
        X_val, y_val, y_eta_val = val["X"], val["y"], val["y_eta"]
        X_test, y_test, y_eta_test = test["X"], test["y"], test["y_eta"]

        combined_X = np.concatenate([X_train, X_val, X_test])
        combined_y_eta = np.concatenate([y_eta_train, y_eta_val, y_eta_test])
        combined_y = np.concatenate([y_train, y_val, y_test])
        
        print(f"Found {len(combined_y)} events")

        # Obtain true mass from ROOT file
        root_file = ROOT.TFile.Open(os.path.join(DATA_DIRECTORY, name))
        test_tree = root_file.Get("test")
        test_tree.GetEntry(0)
        true_mass = getattr(test_tree, "T1M")
        root_file.Close()
        
        # Vectorized lorentz addition
        particle_masses = np.zeros((combined_X.shape[0], combined_X.shape[1]))

        full_decay_products = combined_X.copy()
        full_decay_products[:, 3, 1] = combined_y_eta
        calc_mass = vectorized_lorentz_addition(full_decay_products, particle_masses)

        lorentz_add_mass = vectorized_lorentz_addition(combined_X, particle_masses)

        # Error check
        target_y_errors = np.sum(~np.isclose(true_mass, combined_y, rtol=1e-5))
        calc_y_errors = np.sum(~np.isclose(true_mass, calc_mass, rtol=1e-5))
        print(f"{target_y_errors} samples had significant errors in target mass value")
        print(f"{calc_y_errors} samples had significant errors in calculated mass value")

        # Plot for lorentz addition summation
        create_1var_histogram_with_marker(
            data=lorentz_add_mass,
            data_label="Calculated Mass Distribution",
            marker=true_mass,
            marker_label="X True Mass",
            title=f"Mass Regression - Lorentz Addition, {name[:-5]}",
            x_label="Mass (GeV / c^2)",
            filename=f"{OUTPUT_IMAGE_DIRECTORY}/{name}.png"
        )
    
if __name__ == "__main__":
    main()