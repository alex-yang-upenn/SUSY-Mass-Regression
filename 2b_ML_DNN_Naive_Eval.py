import ROOT
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

import os
import pickle
import random

import utils


# Parameters
TRAINING_DIRECTORY = "DNN_checkpoints"
CHECKPOINTS_NAME = "best_model_20241118_103325.keras"
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

        DNNInputs, EtaTarget, MassTarget = [], [], []
        for entryNum in range(0, test_tree.GetEntries()):
            test_tree.GetEntry(entryNum)

            # Retrieve input, output, and true mass values
            features = utils.single_entry_extract_features(
                tree=test_tree,
                branches=TARGET_BRANCHES,
                truthM=truthM,
                truthID=truthID,
                MET_ids=UNDETECTED_PARTICLES,
                clip_eta=True
            )
            if len(features) > 0:
                DNNInputs.append(features + [getattr(test_tree, "METPt"), getattr(test_tree, "METPhi")])
                EtaTarget.append(getattr(test_tree, "METEta"))
                MassTarget.append(getattr(test_tree, "T1M"))        
        file.Close()
        

        # Load in model and scaler
        checkpoint_path = os.path.join(TRAINING_DIRECTORY, CHECKPOINTS_NAME)
        model = tf.keras.models.load_model(checkpoint_path)
        with open('Preprocessed_Data/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # Pre-process Inputs
        DNNInputs = scaler.transform(np.array(DNNInputs, dtype=np.float32))

        # Compare to true Eta values
        DNNOutputs = model.predict(DNNInputs, verbose=1)
        EtaTarget = np.array(EtaTarget, dtype=np.float32)
        mseEtaDNN = np.mean(np.square(DNNOutputs - EtaTarget))
        print(f"Mean Squared Loss of Eta Predictions: {mseEtaDNN}")

        # Get true Mass values
        trueMass = np.array(MassTarget).mean()
        print(f"True mass: {trueMass}")

        del MassTarget, EtaTarget
        
        # Mass calculations
        NaiveSumPrediction, DNNPrediction = [], []
        unscaledInputs = scaler.inverse_transform(DNNInputs)
        for i in range(len(DNNOutputs)):
            q_1 = utils.create_lorentz_vector(unscaledInputs[i][0], unscaledInputs[i][1], unscaledInputs[i][2], unscaledInputs[i][3])
            q_2 = utils.create_lorentz_vector(unscaledInputs[i][4], unscaledInputs[i][5], unscaledInputs[i][6], unscaledInputs[i][7])
            Lept = utils.create_lorentz_vector(unscaledInputs[i][8], unscaledInputs[i][9], unscaledInputs[i][10], unscaledInputs[i][11])
            
            # Direct Summation, ignoring MET Eta
            NaiveMET = utils.create_lorentz_vector(unscaledInputs[i][12], 0, unscaledInputs[i][13], 0)
            Naive_X_Particle = q_1 + q_2 + Lept + NaiveMET
            NaiveSumPrediction.append(Naive_X_Particle.M())

            # Summation with DNN Eta prediction
            dnnMET = utils.create_lorentz_vector(unscaledInputs[i][12], DNNOutputs[i], unscaledInputs[i][13], 0)
            DNN_X_particle = q_1 + q_2 + Lept + dnnMET
            DNNPrediction.append(DNN_X_particle.M())

        NaiveSumPrediction = np.array(NaiveSumPrediction, dtype=np.float32)
        DNNPrediction = np.array(DNNPrediction, dtype=np.float32)
        mseDNN = np.mean(np.square(DNNPrediction - trueMass))
        mseSum = np.mean(np.square(NaiveSumPrediction - trueMass))
        print(f"Mean Squared Loss of DNN Approach: {mseDNN}")
        print(f"Mean Squared Loss of Naive Summation Approach: {mseSum}")

        # Plot
        utils.create_1var_histogram_with_marker(
            data=DNNPrediction,
            data_label="X Mass Distribution (DNN with neutrino as MET)",
            marker=trueMass,
            marker_label="X True Mass",
            title=f"Mass Regression, {name}",
            x_label="Mass (GeV / c^2)",
            filename=f"{OUTPUT_IMAGE_DIRECTORY}/{name.replace('.root', '')}.png"
        )

if __name__ == "__main__":
    main()