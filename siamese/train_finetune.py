"""
Module Name: train_finetune

Description:
    This module trains a downstream neural network on top of an encoder to predict SUSY particle
    masses, taking advantage of the Contrastive Learning embeddings. For the first few training epochs,
    the encoder weights are finetuned alongside the downstream neural network. The weights are
    frozen for the rest of training.

Usage:
    python train_finetune.py --config config_set2.yaml  # uses config_set2.yaml
Author:
Date:
License:
"""

import os
import sys

import json
import pickle

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from callbacks import get_finetuning_callbacks
from config_loader import load_config
from downstream_model import FinetunedNN
from graph_embeddings import GraphEmbeddings
from simCLR_model import *
from transformation import *
from utils import load_data


def main():
    # Load configuration based on command line argument
    dataset_type = get_dataset_type_from_args()
    config = load_config(dataset_type)

    encoder_dir = os.path.join(SCRIPT_DIR, f"model_{config['RUN_ID']}{config["DATASET_NAME"]}")
    encoder_path = os.path.join(encoder_dir, "best_model_encoder.keras")

    model_dir = os.path.join(SCRIPT_DIR, f"model_{config['RUN_ID']}_finetune{config["DATASET_NAME"]}")
    os.makedirs(model_dir, exist_ok=True)

    # Load in data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        config["PROCESSED_DATA_DIRECTORY"]
    )

    # Create model
    downstream_model = FinetunedNN(
        encoder_path=encoder_path,
        downstream_units=config["DOWNSTREAM_LAYER_SIZES"],
        output_dim=1,
        trainable_encoder=True,
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config["GNN_BASELINE_LEARNING_RATE"]
    )
    downstream_model.compile(optimizer=optimizer, loss="mse", metrics=["mae", "mape"])

    # Build the model with a dummy pass
    _ = downstream_model(X_val)

    # Train
    downstream_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config["EPOCHS"],
        batch_size=config["BATCHSIZE"],
        callbacks=get_finetuning_callbacks(
            model_dir,
            config["DOWNSTREAM_FREEZE_EPOCH"],
            config["DOWNSTREAM_FROZEN_LEARNING_RATE"],
            config["DOWNSTREAM_LEARNING_RATE"],
            config["DOWNSTREAM_LR_DECAY"],
        ),
    )

    test_results = downstream_model.evaluate(X_test, y_test, verbose=1)
    results_dict = {
        "best_model_metrics": {
            "test_mse": float(test_results[0]),
            "test_mae": float(test_results[1]),
            "test_mape": float(test_results[2]),
        }
    }

    with open(os.path.join(model_dir, f"results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)


if __name__ == "__main__":
    main()
