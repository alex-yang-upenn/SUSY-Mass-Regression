import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(ROOT_DIR)

import json
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tqdm import tqdm

import config
from graph_embeddings import GraphEmbeddings
from simCLR_model import *
from transformation import *
from utils import load_data


def main():
    model_dir = os.path.join(SCRIPT_DIR, f"model_{config.RUN_ID}")
    os.makedirs(model_dir, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(config.PROCESSED_DATA_DIRECTORY)

    train_batches = int(len(X_train) // config.BATCHSIZE)
    train_pairs = create_transformed_pairs_dataset(X_train, y_train, config.BATCHSIZE, config.N_FEATURES)

    val_pairs = create_transformed_pairs_dataset(X_val, y_val, config.BATCHSIZE, config.N_FEATURES)
    val_batches = int(len(X_val) / config.BATCHSIZE)

    test_pairs = create_transformed_pairs_dataset(X_test, y_test, config.BATCHSIZE, config.N_FEATURES)
    test_batches = int(len(X_test) // config.BATCHSIZE)

    simclr_model = create_simclr_model(
        n_features=config.N_FEATURES,
        f_r_units=config.GNN_BASELINE_F_R_LAYER_SIZES,
        f_o_units=config.GNN_BASELINE_F_O_LAYER_SIZES,
        phi_C_units=config.SIAMESE_PHI_C_LAYER_SIZES,
        proj_units=config.SIAMESE_PROJ_HEAD_LAYER_SIZES,
        temp=config.SIMCLR_LOSS_TEMP,
        learning_rate=config.GNN_BASELINE_LEARNING_RATE,
    )

    # Call the model once to build it
    for inputs, _ in train_pairs.take(1):
        _ = simclr_model(inputs)
        break

    simclr_model.fit(
        train_pairs,
        validation_data=val_pairs,
        epochs=config.EPOCHS,
        steps_per_epoch=train_batches,
        validation_steps=val_batches,
        callbacks=config.STANDARD_CALLBACKS(model_dir),
    )

    test_loss = simclr_model.evaluate(test_pairs, steps=test_batches, verbose=1)

    results_dict = {
        "contrastive_loss_test_dataset": test_loss
    }

    with open(os.path.join(model_dir, "results.json"), 'w') as f:
        json.dump(results_dict, f, indent=4)

if __name__ == "__main__":
    main()