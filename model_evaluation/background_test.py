"""
Module Name: model_evaluation/background_test

Description:
    Tests all trained models against background data (test_TbqqTblv.npz).
    Creates dual histograms for pairwise model comparisons to analyze how
    different models respond to background events.

Usage:
    python background_test.py

Author:
Date:
License:
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(ROOT_DIR)

import numpy as np
import pickle
import tensorflow as tf
from itertools import combinations
from tqdm import tqdm

import config
from downstream_model import *
from graph_embeddings import GraphEmbeddings
from plotting import create_2var_histogram_with_marker
from simCLR_model import *
from utils import scale_data


def main():
    # Load background data
    data = np.load(os.path.join(config.PROCESSED_DATA_BACKGROUND_DIRECTORY, "test_TbqqTblv.npz"))
    X_background = data['X']

    scalers = []
    for i in range(3):
        with open(os.path.join(config.PROCESSED_DATA_DIRECTORY, f"x_scaler_{i}.pkl"), 'rb') as f:
            scalers.append(pickle.load(f))
    
    X_background_scaled = scale_data(X_background, scalers, [0, 1, 2])
    X_background_scaled = X_background_scaled.transpose(0, 2, 1)

    with open(os.path.join(config.PROCESSED_DATA_DIRECTORY, "y_scaler.pkl"), 'rb') as f:
        y_scaler = pickle.load(f)
    
    
    # Load models
    # GNN baseline model
    gnn_baseline_model_path = os.path.join(config.ROOT_DIR, "gnn_baseline", f"model_{config.RUN_ID}", "best_model.keras")
    gnn_baseline_model = tf.keras.models.load_model(gnn_baseline_model_path)

    y_gnn_baseline_scaled = gnn_baseline_model.predict(X_background_scaled, verbose=1)
    y_gnn_baseline = y_scaler.inverse_transform(y_gnn_baseline_scaled).flatten()
    
    # GNN transformed model
    gnn_transformed_model_path = os.path.join(config.ROOT_DIR, "gnn_transformed", f"model_{config.RUN_ID}", "best_model.keras")
    gnn_transformed_model = tf.keras.models.load_model(gnn_transformed_model_path)
    
    y_gnn_transformed_scaled = gnn_transformed_model.predict(X_background_scaled, verbose=1)
    y_gnn_transformed = y_scaler.inverse_transform(y_gnn_transformed_scaled).flatten()

    # Siamese finetune model
    siamese_finetune_model_path = os.path.join(config.ROOT_DIR, "siamese", f"model_3_finetune", "best_model.keras")
    siamese_finetune_model = tf.keras.models.load_model(
        siamese_finetune_model_path,
        custom_objects={
            "SimCLRNTXentLoss": SimCLRNTXentLoss,
            "GraphEmbeddings": GraphEmbeddings,
            "FinetunedNN": FinetunedNN,
        },
    )

    y_siamese_finetune_scaled = siamese_finetune_model.predict(X_background_scaled, verbose=1)
    y_siamese_finetune = y_scaler.inverse_transform(y_siamese_finetune_scaled).flatten()
    
    # Siamese no finetune model
    siamese_no_finetune_model_path = os.path.join(config.ROOT_DIR, "siamese", f"model_3_no_finetune", "best_model.keras")
    siamese_no_finetune_model = tf.keras.models.load_model(
        siamese_no_finetune_model_path,
        custom_objects={
            "SimCLRNTXentLoss": SimCLRNTXentLoss,
            "GraphEmbeddings": GraphEmbeddings,
            "FinetunedNN": FinetunedNN,
        },
    )

    y_siamese_no_finetune_scaled = siamese_no_finetune_model.predict(X_background_scaled, verbose=1)
    y_siamese_no_finetune = y_scaler.inverse_transform(y_siamese_no_finetune_scaled).flatten()
    
    # Create dual histograms
    create_2var_histogram_with_marker(
        data1=y_gnn_baseline,
        data_label1="GNN Baseline",
        data2=y_siamese_finetune,
        data_label2="Siamese (with finetuning)",
        marker=None,
        marker_label=None,
        title=f"Background Event Predictions",
        x_label="Predicted Mass (GeV / c^2)",
        filename=os.path.join(SCRIPT_DIR, "background_plots/gnn_baseline_v_finetune.png")
    )

    create_2var_histogram_with_marker(
        data1=y_siamese_no_finetune,
        data_label1="Siamese (no finetuning)",
        data2=y_siamese_finetune,
        data_label2="Siamese (with finetuning)",
        marker=None,
        marker_label=None,
        title=f"Background Event Predictions",
        x_label="Predicted Mass (GeV / c^2)",
        filename=os.path.join(SCRIPT_DIR, "background_plots/finetune_v_no_finetune.png")
    )

    create_2var_histogram_with_marker(
        data1=y_gnn_baseline,
        data_label1="GNN Baseline",
        data2=y_gnn_transformed,
        data_label2="GNN Transformed",
        marker=None,
        marker_label=None,
        title=f"Background Event Predictions",
        x_label="Predicted Mass (GeV / c^2)",
        filename=os.path.join(SCRIPT_DIR, "background_plots/transformed_training.png")
    )


if __name__ == "__main__":
    main()