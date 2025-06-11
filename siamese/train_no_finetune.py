"""
Module Name: train_no_finetune

Description:
    This module trains a downstream neural network on top of an encoder to predict SUSY particle
    masses, taking advantage of the Contrastive Learning embeddings. The encoder is frozen for all
    of training, only the downstream network changes.

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

import config
from downstream_model import FinetunedNN
from graph_embeddings import GraphEmbeddings
from simCLR_model import *
from transformation import *
from utils import load_data


def main():
    encoder_dir = os.path.join(SCRIPT_DIR, f"model_{config.RUN_ID}")
    encoder_path = os.path.join(encoder_dir, "best_model_encoder.keras")

    model_dir = os.path.join(SCRIPT_DIR, f"model_{config.RUN_ID}_no_finetune")
    os.makedirs(model_dir, exist_ok=True)
    
    # Load in data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(config.PROCESSED_DATA_DIRECTORY)
    
    # Create model
    downstream_model = FinetunedNN(
        encoder_path=encoder_path,
        downstream_units=config.SIAMESE_DOWNSTREAM_LAYER_SIZES,
        output_dim=1,
        trainable_encoder=False,  # Freeze encoder weights
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.DOWNSTREAM_NO_FINETUNE_LEARNING_RATE)
    downstream_model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mape']
    )

    # Build the model with a dummy pass
    _ = downstream_model(X_val)
    
    # Train
    downstream_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCHSIZE,
        callbacks=config.STANDARD_CALLBACKS(model_dir),
    )

    test_results = downstream_model.evaluate(X_test, y_test, verbose=1)
    results_dict = {
        "best_model_metrics": {
            "test_mse": float(test_results[0]),
            "test_mae": float(test_results[1]),
            "test_mape": float(test_results[2]),
        }
    }

    with open(os.path.join(model_dir, f"results.json"), 'w') as f:
        json.dump(results_dict, f, indent=4)
    
if __name__ == "__main__":
    main()