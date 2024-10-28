import ROOT
import tensorflow as tf
import numpy as np

import os
import random
import sys

import utils


# Parameters
TRAINING_DIRECTORY = "DNN_checkpoints"
CHECKPOINTS_NAME = "best_model_20241028_110340.keras"
DATA_DIRECTORY = "MassRegressionData"
OUTPUT_IMAGE_DIRECTORY = "DNN_Naive_Graphs"
TARGET_BRANCHES = ["P1"]
UNDETECTED_PARTICLES = [12]

os.makedirs(OUTPUT_IMAGE_DIRECTORY, exist_ok=True)

random.seed(42)


def main():
    rootFiles = random.sample(os.listdir(DATA_DIRECTORY), 5)
    for name in rootFiles:
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

        DNNInputs = []
        MassTarget = []
        for entryNum in range(0, test_tree.GetEntries()):
            test_tree.GetEntry(entryNum)
            DNNInputs.append(utils.extract_features_particles_list(
                tree=test_tree,
                branches=TARGET_BRANCHES,
                truthM=truthM,
                truthID=truthID,
                MET_ids=UNDETECTED_PARTICLES
            ) + [getattr(test_tree, "METPt"), getattr(test_tree, "METPhi")])
            
            MassTarget.append(getattr(test_tree, "T1M"))
        
        # Close file
        file.Close()

        # Load in model
        checkpoint_path = os.path.join(TRAINING_DIRECTORY, CHECKPOINTS_NAME)
        model = tf.keras.models.load_model(checkpoint_path)

        # Get DNN outputs
        DNNInputs = np.array(DNNInputs)
        DNNOutputs = model.predict(DNNInputs, verbose=1)

        # Plot
        utils.create_1var_histogram_with_marker(
            data=DNNOutputs,
            data_label="X Mass Distribution (DNN with neutrino as MET)",
            marker=np.array(MassTarget).mean(),
            marker_label="X True Mass",
            title=f"Mass Regression, {name}",
            x_label="Mass (GeV / c^2)",
            filename=f"{OUTPUT_IMAGE_DIRECTORY}/{name.replace('.root', '')}.png"
        )

if __name__ == "__main__":
    main()