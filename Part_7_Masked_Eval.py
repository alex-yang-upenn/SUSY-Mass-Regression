import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import ROOT
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

import json
import pickle
import random
import re

import ROOT_utils
from tf_utils import GraphEmbeddings
import utils


# Parameters
DATA_DIRECTORY = "raw_data"
TARGET_BRANCHES = ["P1"]
UNDETECTED_PARTICLES = [12]

DNN_TRAINING_DIRECTORY = "DNN_Checkpoints"
DNN_MODEL_CHECKPOINT = "best_model_12-23_22:28.keras"
DNN_IMAGE_DIRECTORY = "DNN_Masked_Graphs"

GNN_TRAINING_DIRECTORY = "GNN_Checkpoints"
GNN_MODEL_CHECKPOINT = "best_model_12-24_22:03.keras"
GNN_IMAGE_DIRECTORY = "GNN_Masked_Graphs"

os.makedirs(DNN_IMAGE_DIRECTORY, exist_ok=True)
os.makedirs(GNN_IMAGE_DIRECTORY, exist_ok=True)

random.seed(42)


def DNN_Transform(x):
    transformed_x = x.copy()
    transformed_x[:, 4:8] = 0.0
    return transformed_x

def GNN_Transform(x):
    transformed_x = utils.reformat_data(x)
    transformed_x = np.delete(transformed_x, 1, axis=2)
    return transformed_x

def main():
    rootFiles = random.sample(os.listdir(DATA_DIRECTORY), 5)

    # Load in scaler
    with open("pre-processed_data/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # Load in models
    eval_models = {
        "DNN": {
            "model": tf.keras.models.load_model(
                os.path.join(DNN_TRAINING_DIRECTORY, DNN_MODEL_CHECKPOINT)
            ),
            "input_transform": DNN_Transform,
            "image_directory": DNN_IMAGE_DIRECTORY
        },
        "GNN": {
            "model": tf.keras.models.load_model(
                os.path.join(GNN_TRAINING_DIRECTORY, GNN_MODEL_CHECKPOINT),
                custom_objects={"GraphEmbeddings": GraphEmbeddings}
            ),
            "input_transform": GNN_Transform,
            "image_directory": GNN_IMAGE_DIRECTORY
        }
    }
    results = {}

    for filepath in rootFiles:
        print(f"Processing file {filepath}")
        file = ROOT.TFile.Open(os.path.join(DATA_DIRECTORY, filepath))

        # Extract data
        info_tree = file.Get("info")
        info_tree.GetEntry(0)
        truthM = list(getattr(info_tree, "truthM"))
        truthID = list(getattr(info_tree, "truthId"))

        test_tree = file.Get("test")
        inputs, etaTarget = [], []
        for entryNum in range(0, test_tree.GetEntries()):
            test_tree.GetEntry(entryNum)
            features = ROOT_utils.single_entry_extract_features(
                tree=test_tree,
                branches=TARGET_BRANCHES,
                truthM=truthM,
                truthID=truthID,
                MET_ids=UNDETECTED_PARTICLES,
                clip_eta=True
            )
            if len(features) > 0:
                inputs.append(features + [getattr(test_tree, "METPt"), getattr(test_tree, "METPhi")])
                etaTarget.append(getattr(test_tree, "METEta"))

        trueMass = np.float32(getattr(test_tree, "T1M"))

        file.Close()
        del truthM, truthID, 

        # Model input and output
        unscaledInputs = np.array(inputs, dtype=np.float32)
        inputs = scaler.transform(unscaledInputs)
        etaTarget = np.array(etaTarget, dtype=np.float32)

        # Eta calculations
        outputs = {}
        for name, model in eval_models.items():
            model_inputs = model["input_transform"](inputs)
            outputs[name + "_eta"] = model["model"].predict(model_inputs, verbose=1).flatten()
            outputs[name + "_mass"] = []
        
        # Mass calculations
        unscaledInputs = scaler.inverse_transform(inputs)
        naive_mass = []
        for i in range(len(unscaledInputs)):
            q_1 = ROOT_utils.create_lorentz_vector(unscaledInputs[i][0], unscaledInputs[i][1], unscaledInputs[i][2], unscaledInputs[i][3])
            Lept = ROOT_utils.create_lorentz_vector(unscaledInputs[i][8], unscaledInputs[i][9], unscaledInputs[i][10], unscaledInputs[i][11])
            NaiveMET = ROOT_utils.create_lorentz_vector(unscaledInputs[i][12], 0, unscaledInputs[i][13], 0)

            Naive_X_Particle = q_1 + Lept + NaiveMET
            naive_mass.append(Naive_X_Particle.M())

            for name, model in eval_models.items():
                met = ROOT_utils.create_lorentz_vector(unscaledInputs[i][12], outputs[name + "_eta"][i], unscaledInputs[i][13], 0)
                X_particle = q_1 + Lept + met
                outputs[name + "_mass"].append(X_particle.M())

        # Logging and displaying results
        filename = re.search("X[0-9]*_Y[0-9]*\.", filepath).group(0)[:-1]
        currFileResults = {}
        currFileResults["True Mass"] = float(trueMass)

        naive_mass = np.array(naive_mass, dtype=np.float32)
        naiveMseEta = np.mean(np.square(etaTarget))
        naiveMseMass = np.mean(np.square(naive_mass - trueMass))
        naiveMapeMass = np.mean(np.abs((trueMass - naive_mass) / trueMass)) * 100.0
        currFileResults["Naive Sum"] = {}
        currFileResults["Naive Sum"]["Eta_MSE"] = float(naiveMseEta)
        currFileResults["Naive Sum"]["Mass_MSE"] = float(naiveMseMass)
        currFileResults["Naive Sum"]["Mass_MAPE"] = float(naiveMapeMass)

        for name, model in eval_models.items():
            currFileResults[name] = {} 
            
            outputs[name + "_eta"] = np.array(outputs[name + "_eta"], dtype=np.float32)
            mseEta = np.mean(np.square(outputs[name + "_eta"] - etaTarget))
            currFileResults[name]["Eta_MSE"] = float(mseEta)

            outputs[name + "_mass"] = np.array(outputs[name + "_mass"], dtype=np.float32)
            mseMass = np.mean(np.square(outputs[name + "_mass"] - trueMass))
            mapeMass = np.mean(np.abs((trueMass - outputs[name + "_mass"]) / trueMass)) * 100.0
            currFileResults[name]["Mass_MSE"] = float(mseMass)
            currFileResults[name]["Mass_MAPE"] = float(mapeMass)

            utils.create_2var_histogram_with_marker(
                data1=outputs[name + "_mass"],
                data_label1="X Mass ({})".format(name),
                data2=naive_mass,
                data_label2="X Mass (Naive Summation)",
                marker=trueMass,
                marker_label="X True Mass",
                title=f"X Mass Regression Distribution (Neutrino + Quark 2 as MET), {filepath}",
                x_label="Mass (GeV / c^2)",
                filename="{}/{}.png".format(model["image_directory"], filename)
            )
        results[filename] = currFileResults
    
    with open(f"model_evaluation_masked.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()