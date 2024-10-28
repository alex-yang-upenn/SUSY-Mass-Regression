"""
Module Name: example_module

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

import os
import sys
import random

import utils


# Parameters
DATA_DIRECTORY = "MassRegressionData"
TRAIN_DATA_DIRECTORY = "Preprocessed_Data/Train"
TEST_DATA_DIRECTORY = "Preprocessed_Data/Test"
TRAIN_TEST_SPLIT = 0.2
CROSS_VALIDATION_SECTION = 1
TARGET_BRANCHES = ["P1"]
UNDETECTED_PARTICLES = [12]
CHUNK_SIZE = 15000

if CROSS_VALIDATION_SECTION * TRAIN_TEST_SPLIT > 1:
    raise AssertionError("Mismatch in CROSS_VALIDATAION_SECTION and TRAIN_TEST_SPLIT")

os.makedirs(TRAIN_DATA_DIRECTORY, exist_ok=True)
os.makedirs(TEST_DATA_DIRECTORY, exist_ok=True)


def main():
    trainIndex = 1
    testIndex = 1
    TrainInputs = []
    TrainOutputs = []
    TestInputs = []
    TestOutputs = []
    for name in os.listdir(DATA_DIRECTORY):
        if name[-5:] != ".root":
            continue

        # Access ROOT file
        print(f"Processing file {name}")
        filename = os.path.join(DATA_DIRECTORY, name)
        file = ROOT.TFile.Open(filename)

        # Collect "info" branch
        info_tree = file.Get("info")
        info_tree.GetEntry(0)
        truthM = list(getattr(info_tree, "truthM"))
        truthID = list(getattr(info_tree, "truthId"))

        # Iterate through "test" branch
        test_tree = file.Get("test")

        sample_count = test_tree.GetEntries()
        cross_val_section_start = (CROSS_VALIDATION_SECTION - 1) * TRAIN_TEST_SPLIT * sample_count
        cross_val_section_end = CROSS_VALIDATION_SECTION * TRAIN_TEST_SPLIT * sample_count

        for entryNum in range(0, sample_count):
            test_tree.GetEntry(entryNum)
            if entryNum >= cross_val_section_start and entryNum < cross_val_section_end:
                TestInputs.append(utils.extract_features_particles_list(
                    tree=test_tree,
                    branches=TARGET_BRANCHES,
                    truthM=truthM,
                    truthID=truthID,
                    MET_ids=UNDETECTED_PARTICLES
                ) + [getattr(test_tree, "METPt"), getattr(test_tree, "METPhi")])
                TestOutputs.append(getattr(test_tree, "T1M"))

                if len(TestOutputs) == CHUNK_SIZE:
                    savedTestInputs = np.array(TestInputs)
                    savedTestOutputs = np.array(TestOutputs)
                    np.save(os.path.join(TEST_DATA_DIRECTORY, f"test_input_{testIndex}"), savedTestInputs)
                    np.save(os.path.join(TEST_DATA_DIRECTORY, f"test_output_{testIndex}"), savedTestOutputs)
                    TestInputs.clear()
                    TestOutputs.clear()
                    testIndex += 1
            else:
                TrainInputs.append(utils.extract_features_particles_list(
                    tree=test_tree,
                    branches=TARGET_BRANCHES,
                    truthM=truthM,
                    truthID=truthID,
                    MET_ids=UNDETECTED_PARTICLES
                ) + [getattr(test_tree, "METPt"), getattr(test_tree, "METPhi")])
                TrainOutputs.append(getattr(test_tree, "T1M"))

                if len(TrainOutputs) == CHUNK_SIZE:
                    savedTrainInputs = np.array(TrainInputs)
                    savedTrainOutputs = np.array(TrainOutputs)
                    np.save(os.path.join(TRAIN_DATA_DIRECTORY, f"train_input_{trainIndex}"), savedTrainInputs)
                    np.save(os.path.join(TRAIN_DATA_DIRECTORY, f"train_output_{trainIndex}"), savedTrainOutputs)
                    TrainInputs.clear()
                    TrainOutputs.clear()
                    trainIndex += 1

        file.Close()

    np.save(os.path.join(TRAIN_DATA_DIRECTORY, f"train_input_{trainIndex}"), savedTrainInputs)
    np.save(os.path.join(TRAIN_DATA_DIRECTORY, f"train_output_{trainIndex}"), savedTrainOutputs)
    
    print(f"Created {trainIndex} chunks for training")
    print(f"Created {testIndex} chunks for testing")

if __name__ == "__main__":
    main()