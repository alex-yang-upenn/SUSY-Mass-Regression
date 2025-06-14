"""
Module Name: model_comparison/full_dataset

Description:
    Identical evaluation as full_dataset, a comparison of models' performance on the full
    test dataset. However, this module first applies an augmentation to the input data.

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

import config
from downstream_model import *
from graph_embeddings import GraphEmbeddings
from simCLR_model import *
from transformation import create_transformed_dataset, ViewTransformedGenerator
from utils import *
from plotting import *


def main():
    # Predictions given by gnn_baseline model
    _, _, _, _, X, y = load_data(config.PROCESSED_DATA_DIRECTORY)

    test_size = len(X)
    test_batches = int(test_size // config.BATCHSIZE)

    test_transformed = create_transformed_dataset(X, y, batchsize=config.BATCHSIZE, n_features=config.N_FEATURES)
    test_transformed = test_transformed.take(test_batches)

    gnn_baseline_model_path = os.path.join(config.ROOT_DIR, "gnn_baseline", f"model_{config.RUN_ID}", "best_model.keras")
    gnn_baseline_model = tf.keras.models.load_model(gnn_baseline_model_path)

    with open(os.path.join(config.PROCESSED_DATA_DIRECTORY, "y_scaler.pkl"), 'rb') as f:
        y_scaler = pickle.load(f)
    
    # Predictions given by gnn_baseline model
    y_gnn_baseline_list = []
    y_true_list = []
    for features, labels in tqdm(test_transformed, desc="GNN Baseline", total=test_batches):
        y_gnn_baseline_batch_scaled = gnn_baseline_model.predict(features, verbose=0)
        y_gnn_baseline_batch = y_scaler.inverse_transform(y_gnn_baseline_batch_scaled)
        y_gnn_baseline_list.append(y_gnn_baseline_batch)
        y_true_batch = y_scaler.inverse_transform(labels.numpy())
        y_true_list.append(y_true_batch)
    y_gnn_baseline = np.concatenate(y_gnn_baseline_list).flatten()
    y_true = np.concatenate(y_true_list).flatten()

    # Predictions given by gnn_transformed model
    gnn_transformed_model_path = os.path.join(config.ROOT_DIR, "gnn_transformed", f"model_{config.RUN_ID}", "best_model.keras")
    gnn_transformed_model = tf.keras.models.load_model(gnn_transformed_model_path)
    
    y_gnn_transformed_list = []
    y_true_list = []
    for features, labels in tqdm(test_transformed, desc="GNN Transformed", total=test_batches):
        y_gnn_transformed_batch_scaled = gnn_transformed_model.predict(features, verbose=0)
        y_gnn_transformed_batch = y_scaler.inverse_transform(y_gnn_transformed_batch_scaled)
        y_gnn_transformed_list.append(y_gnn_transformed_batch)
        y_true_batch = y_scaler.inverse_transform(labels.numpy())
        y_true_list.append(y_true_batch)
    y_gnn_transformed = np.concatenate(y_gnn_transformed_list).flatten()
    y_true = np.concatenate(y_true_list).flatten()

    # Predictions given by siamese model
    siamese_model_path = os.path.join(config.ROOT_DIR, "siamese", f"model_{config.RUN_ID}_downstream", "best_model.keras")
    siamese_model = tf.keras.models.load_model(
        siamese_model_path,
        custom_objects={
            "SimCLRNTXentLoss": SimCLRNTXentLoss,
            "GraphEmbeddings": GraphEmbeddings,
            "FinetunedNN": FinetunedNN,
        },
    )

    y_siamese_list = []
    y_true_list = []
    for features, labels in tqdm(test_transformed, desc="GNN Transformed", total=test_batches):
        y_siamese_batch_scaled = siamese_model.predict(features, verbose=0)
        y_siamese_batch = y_scaler.inverse_transform(y_siamese_batch_scaled)
        y_siamese_list.append(y_siamese_batch)
        y_true_batch = y_scaler.inverse_transform(labels.numpy())
        y_true_list.append(y_true_batch)
    y_siamese = np.concatenate(y_siamese_list).flatten()
    y_true = np.concatenate(y_true_list).flatten()

    # Predictions given by naive lorentz addition
    _, _, _, _, X_orig, y_orig = load_data_original(config.PROCESSED_DATA_DIRECTORY)
    
    X_orig = X_orig.transpose(0, 2, 1)
    y_orig = y_orig.reshape(-1, 1)

    test_orig_transformed = create_transformed_dataset(X_orig, y_orig, batchsize=config.BATCHSIZE, n_features=config.N_FEATURES)
    test_orig_transformed = test_orig_transformed.take(test_batches)

    y_lorentz_list = []
    y_lorentz_true_list = []
    for features, labels in tqdm(test_orig_transformed, desc="Lorentz Addition", total=test_batches):
        particles = features.numpy().transpose(0, 2, 1)
        particle_masses = np.zeros((particles.shape[0], particles.shape[1]))
        y = vectorized_lorentz_addition(particles, particle_masses)
        y_lorentz_list.append(y)
        y_lorentz_true_list.append(labels.numpy())
    y_lorentz = np.concatenate(y_lorentz_list).flatten()
    y_lorentz_true = np.concatenate(y_lorentz_true_list).flatten()

    baseline_model_metrics = calculate_metrics(y_true, y_gnn_baseline, "gnn_baseline")
    transformed_model_metrics = calculate_metrics(y_true, y_gnn_transformed, "gnn_transformed")
    siamese_model_metrics = calculate_metrics(y_true, y_siamese, "siamese")
    lorentz_metrics = calculate_metrics(y_lorentz_true, y_lorentz, "lorentz_addition")

    metrics = {
        **baseline_model_metrics,
        **transformed_model_metrics,
        **siamese_model_metrics,
        **lorentz_metrics
    }
    with open(os.path.join(SCRIPT_DIR, "transformed_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()