"""
Module Name: gnn_transformed

Description:
    This module trains and evaluates the same GNN model as the baseline. However, the training
    dataset is randomly augmented with transformations that distort the particle data.

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

import json
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tqdm import tqdm

import config_set2
from graph_embeddings import GraphEmbeddings
from transformation import create_transformed_dataset, ViewTransformedGenerator
from utils import load_data


def build_model_helper():
    """
    Each sample has shape (N_FEATURES, N_PARTICLES)
    """
    input = tf.keras.Input(shape=(config_set2.N_FEATURES, None), dtype=tf.float32)

    # Reduce graph to vector embeddings
    O_Bar = GraphEmbeddings(
        f_r_units=config_set2.GNN_BASELINE_F_R_LAYER_SIZES,
        f_o_units=config_set2.GNN_BASELINE_F_O_LAYER_SIZES,
    )(input)
    
    # Trainable function phi_C to compute MET Eta from vector embeddings
    phi_C = tf.keras.Sequential([
        layer
        for units in config_set2.GNN_BASELINE_PHI_C_LAYER_SIZES
        for layer in [
            tf.keras.layers.Dense(units),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ]
    ])(O_Bar)

    output = tf.keras.layers.Dense(1)(phi_C)
    
    # Create and compile model
    model = tf.keras.Model(inputs=input, outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config_set2.GNN_TRANSFORMED_LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    return model

def main():
    model_dir = os.path.join(SCRIPT_DIR, f"model_{config_set2.RUN_ID}_set2")
    os.makedirs(model_dir, exist_ok=True)

    (
        X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled
    ) = load_data(config_set2.PROCESSED_DATA_DIRECTORY)
    train_batches = int(len(y_train_scaled) // config_set2.BATCHSIZE)
    train_transformed = create_transformed_dataset(
        X_train_scaled, y_train_scaled, batchsize=config_set2.BATCHSIZE, n_features=config_set2.N_FEATURES
    )
    val_batches = int(len(y_val_scaled) / config_set2.BATCHSIZE)
    val_transformed = create_transformed_dataset(
        X_val_scaled, y_val_scaled, batchsize=config_set2.BATCHSIZE, n_features=config_set2.N_FEATURES
    )
    test_batches = int(len(y_test_scaled) // config_set2.BATCHSIZE)
    test_transformed = create_transformed_dataset(
        X_test_scaled, y_test_scaled, batchsize=config_set2.BATCHSIZE, n_features=config_set2.N_FEATURES
    )

    model = build_model_helper()

    model.fit(
        train_transformed,
        validation_data=val_transformed,
        epochs=config_set2.EPOCHS,
        steps_per_epoch=train_batches,
        validation_steps=val_batches,
        callbacks=config_set2.NO_STOP_CALLBACKS(model_dir),
    )

    test_results = model.evaluate(test_transformed, steps=test_batches, verbose=1)

    results_dict = {
        "best_model_metrics": {
            "test_mse": float(test_results[0]),
            "test_mae": float(test_results[1]),
            "test_mape": float(test_results[2]),
        }
    }

    with open(os.path.join(model_dir, "results.json"), 'w') as f:
        json.dump(results_dict, f, indent=4)


if __name__ == "__main__":
    main()