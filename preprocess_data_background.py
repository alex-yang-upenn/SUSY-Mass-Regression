"""
Module Name: preprocess_data_background

Description:
    Reads the data from the background event ROOT file, extracts information, and reformats it to look like
    an input to the models. Stores the result in numpy files.

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

import os

import config
import numpy as np
import ROOT

from ROOT_utils import extract_event_features

os.makedirs(config.PROCESSED_DATA_BACKGROUND_DIRECTORY, exist_ok=True)

BACKGROUND_FILE = "test_TbqqTblv.root"


def main():
    file = ROOT.TFile.Open(
        os.path.join(config.RAW_DATA_BACKGROUND_DIRECTORY, BACKGROUND_FILE)
    )
    X = []

    test_tree = file.Get("test")
    sample_count = test_tree.GetEntries()

    for entryNum in range(sample_count):
        test_tree.GetEntry(entryNum)
        features_1 = extract_event_features(
            tree=test_tree,
            decay_chain="P1",
            MET_ids=config.MET_IDS,
            Lepton_ids=config.LEPTON_IDS,
        )
        features_2 = extract_event_features(
            tree=test_tree,
            decay_chain="P2",
            MET_ids=config.MET_IDS,
            Lepton_ids=config.LEPTON_IDS,
        )

        if features_1 and features_2:
            X.append([features_1[1], features_1[2], features_2[1], features_2[2]])

    X = np.array(X, dtype=np.float32)
    np_filename = BACKGROUND_FILE[0:-5] + ".npz"
    np.savez_compressed(
        os.path.join(config.PROCESSED_DATA_BACKGROUND_DIRECTORY, np_filename), X=X
    )

    file.Close()


if __name__ == "__main__":
    main()
