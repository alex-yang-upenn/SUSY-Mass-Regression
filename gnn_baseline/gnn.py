"""
Module Name: gnn

Description:
    This module trains and evaluates a supervised learning model to predict the mass of intermediate SUSY particles.
    The architecture is a standard GNN, with dense layers afterwards that compute a single float value.

Usage:
Author:
Date:
License:
"""
import os

import json
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tqdm import tqdm

from config_loader import load_config
from callbacks import get_standard_callbacks
from graph_embeddings import GraphEmbeddings
from utils import load_data_original, normalize_data, scale_data


def build_model_helper(config):
    """
    Each sample has shape (N_FEATURES, N_PARTICLES)
    """
    input = tf.keras.Input(shape=(config['N_FEATURES'], None), dtype=tf.float32)

    # Reduce graph to vector embeddings
    O_Bar = GraphEmbeddings(
        f_r_units=config['GNN_BASELINE_F_R_LAYER_SIZES'],
        f_o_units=config['GNN_BASELINE_F_O_LAYER_SIZES'],
    )(input)
    
    # Trainable function phi_C to compute MET Eta from vector embeddings
    phi_C = tf.keras.Sequential([
        layer
        for units in config['GNN_BASELINE_PHI_C_LAYER_SIZES']
        for layer in [
            tf.keras.layers.Dense(units),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ]
    ])(O_Bar)

    output = tf.keras.layers.Dense(1)(phi_C)
    
    # Create and compile model
    model = tf.keras.Model(inputs=input, outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['GNN_BASELINE_LEARNING_RATE'])
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    return model

def main():
    config = load_config()
    
    model_dir = os.path.join(
        f"{config["ROOT_DIR"]}",
        "gnn_baseline",
        f"model_{config['RUN_ID']}{config["DATASET_NAME"]}"
    )
    os.makedirs(model_dir, exist_ok=True)
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_original(config['PROCESSED_DATA_DIRECTORY'])

    X_train_scaled, scalers = normalize_data(X_train, [0, 1, 2])
    X_val_scaled = scale_data(X_val, scalers, [0, 1, 2])
    X_test_scaled = scale_data(X_test, scalers, [0, 1, 2])
    
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

    for i, s in enumerate(scalers):
        with open(os.path.join(config['PROCESSED_DATA_DIRECTORY'], f"x_scaler_{i}.pkl"), 'wb') as f:
            pickle.dump(s, f)

    with open(os.path.join(config['PROCESSED_DATA_DIRECTORY'], "y_scaler.pkl"), 'wb') as f:
        pickle.dump(y_scaler, f)

    del X_train, X_val, X_test, y_train, y_val, y_test
    del scalers, y_scaler

    # Change from (batchsize, num particles, num features) to (batchsize, num features, num particles)
    X_train_scaled = X_train_scaled.transpose(0, 2, 1)
    X_val_scaled = X_val_scaled.transpose(0, 2, 1)
    X_test_scaled = X_test_scaled.transpose(0, 2, 1)

    model = build_model_helper(config)

    model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=config['EPOCHS'],
        batch_size=config['BATCHSIZE'],
        callbacks=get_standard_callbacks(
            model_dir,
            config['GNN_BASELINE_EARLY_STOPPING_PATIENCE'],
            config['GNN_BASELINE_REDUCE_LR_PATIENCE']
        ),
    )

    test_results = model.evaluate(X_test_scaled, y_test_scaled, verbose=1)

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