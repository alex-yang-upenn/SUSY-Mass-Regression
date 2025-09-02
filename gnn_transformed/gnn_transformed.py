"""
Module Name: gnn_transformed

Description:
    This module trains and evaluates the same GNN model as the baseline. However, the training
    dataset is randomly augmented with transformations that distort the particle data.

Usage:
    python gnn_transformed.py --config config_set2.yaml
Author:
Date:
License:
"""

import json
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from callbacks import get_no_stop_callbacks
from config_loader import load_config
from graph_embeddings import GraphEmbeddings
from transformation import create_transformed_dataset
from utils import load_data


def build_model_helper(config):
    """
    Each sample has shape (N_FEATURES, N_PARTICLES)
    """
    input = tf.keras.Input(shape=(config["N_FEATURES"], None), dtype=tf.float32)

    # Reduce graph to vector embeddings
    O_Bar = GraphEmbeddings(
        f_r_units=config["GNN_BASELINE_F_R_LAYER_SIZES"],
        f_o_units=config["GNN_BASELINE_F_O_LAYER_SIZES"],
    )(input)

    # Trainable function phi_C to compute MET Eta from vector embeddings
    phi_C = tf.keras.Sequential(
        [
            layer
            for units in config["GNN_BASELINE_PHI_C_LAYER_SIZES"]
            for layer in [
                tf.keras.layers.Dense(units),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ]
        ]
    )(O_Bar)

    output = tf.keras.layers.Dense(1)(phi_C)

    # Create and compile model
    model = tf.keras.Model(inputs=input, outputs=output)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config["GNN_TRANSFORMED_LEARNING_RATE"]
    )
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae", "mape"])

    return model


def main():
    config = load_config()

    model_dir = os.path.join(
        config["ROOT_DIR"],
        "gnn_transformed",
        f"model_{config['RUN_ID']}{config["DATASET_NAME"]}",
    )
    os.makedirs(model_dir, exist_ok=True)

    (
        X_train_scaled,
        y_train_scaled,
        X_val_scaled,
        y_val_scaled,
        X_test_scaled,
        y_test_scaled,
    ) = load_data(config["PROCESSED_DATA_DIRECTORY"])
    train_batches = int(len(y_train_scaled) // config["BATCHSIZE"])
    train_transformed = create_transformed_dataset(
        X_train_scaled,
        y_train_scaled,
        batchsize=config["BATCHSIZE"],
        n_features=config["N_FEATURES"],
    )
    val_batches = int(len(y_val_scaled) / config["BATCHSIZE"])
    val_transformed = create_transformed_dataset(
        X_val_scaled,
        y_val_scaled,
        batchsize=config["BATCHSIZE"],
        n_features=config["N_FEATURES"],
    )
    test_batches = int(len(y_test_scaled) // config["BATCHSIZE"])
    test_transformed = create_transformed_dataset(
        X_test_scaled,
        y_test_scaled,
        batchsize=config["BATCHSIZE"],
        n_features=config["N_FEATURES"],
    )

    model = build_model_helper(config)

    model.fit(
        train_transformed,
        validation_data=val_transformed,
        epochs=config["EPOCHS"],
        steps_per_epoch=train_batches,
        validation_steps=val_batches,
        callbacks=get_no_stop_callbacks(
            model_dir, config["GNN_TRANSFORMED_LEARNING_RATE_DECAY"]
        ),
    )

    test_results = model.evaluate(test_transformed, steps=test_batches, verbose=1)

    results_dict = {
        "best_model_metrics": {
            "test_mse": float(test_results[0]),
            "test_mae": float(test_results[1]),
            "test_mape": float(test_results[2]),
        }
    }

    with open(os.path.join(model_dir, "results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)


if __name__ == "__main__":
    main()
