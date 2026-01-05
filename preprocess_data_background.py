"""Preprocessing script for SUSY background event data.

This module converts the background event ROOT file (test_TbqqTblv.root) into preprocessed
NumPy arrays formatted for compatibility with the signal event models.

Background events represent non-SUSY processes and are used for model evaluation.
The script extracts features from two decay chains (P1 and P2) per event, selecting
specific particles (lepton and second particle from each chain) to match the signal
data format. Output shape is (N_EVENTS, 4, N_FEATURES) where 4 particles are selected
and features include pt, eta, phi, and one-hot encoded particle types (MET, Lepton, Other).

Typical usage example:
    python preprocess_data_background.py
"""

import os

import config
import numpy as np
import ROOT

from ROOT_utils import extract_event_features

os.makedirs(config.PROCESSED_DATA_BACKGROUND_DIRECTORY, exist_ok=True)

BACKGROUND_FILE = "test_TbqqTblv.root"


def main():
    """Process raw background ROOT data into NumPy format for model evaluation.

    Extracts features from the background event file (test_TbqqTblv.root) by processing
    two decay chains per event. Selects 4 particles total (2 from each chain: lepton
    and second particle) to create feature arrays compatible with signal data models.

    The function:
        1. Opens the background ROOT file from configured directory
        2. Iterates through all events in the 'test' tree
        3. Extracts features from P1 and P2 decay chains
        4. Selects specific particles (indices 1 and 2) from each chain
        5. Saves compressed .npz file with X (features) only

    Note:
        Background data has no target masses, only features are saved.
    """
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
