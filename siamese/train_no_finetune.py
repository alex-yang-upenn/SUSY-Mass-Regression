"""Frozen encoder training for mass regression.

Trains a downstream regression head on top of the pretrained SimCLR encoder
with frozen encoder weights. This approach uses the pretrained features as-is
without adaptation, testing the quality of the learned representations directly.

Usage:
    python3 -m siamese.train_no_finetune                    # Uses config.yaml
    python3 -m siamese.train_no_finetune --config set2      # Uses config_set2.yaml
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(ROOT_DIR)

import json
import pickle

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from callbacks import get_standard_callbacks
from config_loader import get_dataset_type_from_args, load_config
from downstream_model import FinetunedNN
from graph_embeddings import GraphEmbeddings
from simCLR_model import *
from transformation import *
from utils import load_data


def main():
    # Load configuration based on command line argument
    dataset_type = get_dataset_type_from_args()
    config = load_config(dataset_type)

    encoder_dir = os.path.join(SCRIPT_DIR, f"model_{config['RUN_ID']}")
    encoder_path = os.path.join(encoder_dir, "best_model_encoder.keras")

    model_dir = os.path.join(SCRIPT_DIR, f"model_{config['RUN_ID']}_no_finetune")
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
        trainable_encoder=False,  # Freeze encoder weights
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config["DOWNSTREAM_NO_FINETUNE_LEARNING_RATE"]
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
        callbacks=get_standard_callbacks(
            model_dir,
            config["GNN_BASELINE_EARLY_STOPPING_PATIENCE"],
            config["GNN_BASELINE_REDUCE_LR_PATIENCE"],
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
