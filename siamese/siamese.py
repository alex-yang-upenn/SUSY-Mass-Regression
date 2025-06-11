"""
Module Name: siamese

Description:
    This module trains and evaluates a Contrastive Learning siamese model to encode particle
    data from decay chains. The embeddings will then be used to predict unknown SUSY
    particle masses. At the end of training, the best model is used to generate embeddings for
    train/test/val datasets.

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
from graph_embeddings import GraphEmbeddings
from simCLR_model import *
from transformation import *
from utils import load_data


def main():
    model_dir = os.path.join(SCRIPT_DIR, f"model_{config.RUN_ID}")
    os.makedirs(model_dir, exist_ok=True)

    # Load in data and create augmented pairs
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(config.PROCESSED_DATA_DIRECTORY)

    train_pairs = create_transformed_pairs_dataset(X_train, y_train, config.BATCHSIZE, config.N_FEATURES)
    train_batches = int(len(X_train) // config.BATCHSIZE)

    val_pairs = create_transformed_pairs_dataset(X_val, y_val, config.BATCHSIZE, config.N_FEATURES)
    val_batches = int(len(X_val) / config.BATCHSIZE)

    test_pairs = create_transformed_pairs_dataset(X_test, y_test, config.BATCHSIZE, config.N_FEATURES)
    test_batches = int(len(X_test) // config.BATCHSIZE)

    # Create the model
    simclr_model = create_simclr_model(
        n_features=config.N_FEATURES,
        f_r_units=config.GNN_BASELINE_F_R_LAYER_SIZES,
        f_o_units=config.GNN_BASELINE_F_O_LAYER_SIZES,
        phi_C_units=config.SIAMESE_PHI_C_LAYER_SIZES,
        proj_units=config.SIAMESE_PROJ_HEAD_LAYER_SIZES,
        temp=config.SIMCLR_LOSS_TEMP,
        learning_rate=config.SIAMESE_LEARNING_RATE,
    )

    # Call the model once to build it
    for inputs, _ in train_pairs.take(1):
        _ = simclr_model(inputs)
        break
    
    # Train and evaluate
    simclr_model.fit(
        train_pairs,
        validation_data=val_pairs,
        epochs=config.SIAMESE_EPOCHS,
        steps_per_epoch=train_batches,
        validation_steps=val_batches,
        callbacks=config.STANDARD_CALLBACKS(model_dir, early_stopping_patience=config.SIAMESE_EARLY_STOPPING_PATIENCE),
    )

    train_loss = simclr_model.evaluate(train_pairs, steps=train_batches, verbose=1)
    val_loss = simclr_model.evaluate(val_pairs, steps=val_batches, verbose=1)
    test_loss = simclr_model.evaluate(test_pairs, steps=test_batches, verbose=1)
    results_dict = {
        "train_dataset": {
            "contrastive_loss": train_loss,
        },
        "val_dataset": {
            "contrastive_loss": val_loss,
        },
        "test_dataset": {
            "contrastive_loss": test_loss,
        }
    }

    # Extract encoder and compute embeddings
    encoder_model = simclr_model.encoder
    encoder_model.save(os.path.join(model_dir, f"best_model_encoder.keras"))
    
    # Train embeddings
    train_transformed = create_transformed_dataset(X_train, y_train, config.BATCHSIZE, config.N_FEATURES)
    train_iterator = iter(train_transformed)
    all_train_embeddings, all_train_targets = [], []
    for batch_idx in tqdm(range(train_batches), desc="Computing Train Embeddings"):
        x_batch, y_batch = next(train_iterator)
        batch_embeddings = encoder_model(x_batch)
        all_train_embeddings.append(batch_embeddings.numpy())
        all_train_targets.append(y_batch.numpy())
    train_embeddings = np.concatenate(all_train_embeddings, axis=0)
    train_targets = np.concatenate(all_train_targets, axis=0)

    results_dict["train_dataset"]["embeddings_shape"] = str(train_embeddings.shape)
    results_dict["train_dataset"]["target_shape"] = str(train_targets.shape)
    np.savez_compressed(
        os.path.join(model_dir, "train_embeddings.npz"),
        embeddings=train_embeddings, 
        targets=train_targets,
    )
    del train_transformed, train_iterator, all_train_embeddings, all_train_targets, train_embeddings, train_targets

    # Val Embeddings
    val_transformed = create_transformed_dataset(X_val, y_val, config.BATCHSIZE, config.N_FEATURES)
    val_iterator = iter(val_transformed)
    all_val_embeddings, all_val_targets = [], []
    for batch_idx in tqdm(range(val_batches), desc="Computing Val Embeddings"):
        x_batch, y_batch = next(val_iterator)
        batch_embeddings = encoder_model(x_batch)
        all_val_embeddings.append(batch_embeddings.numpy())
        all_val_targets.append(y_batch.numpy())
    val_embeddings = np.concatenate(all_val_embeddings, axis=0)
    val_targets = np.concatenate(all_val_targets, axis=0)

    results_dict["val_dataset"]["embeddings_shape"] = str(val_embeddings.shape)
    results_dict["val_dataset"]["target_shape"] = str(val_targets.shape)
    np.savez_compressed(
        os.path.join(model_dir, "val_embeddings.npz"),
        embeddings=val_embeddings, 
        targets=val_targets,
    )
    del val_transformed, val_iterator, all_val_embeddings, all_val_targets, val_embeddings, val_targets

    # Test Embeddings
    test_transformed = create_transformed_dataset(X_test, y_test, config.BATCHSIZE, config.N_FEATURES)
    test_iterator = iter(test_transformed)
    all_test_embeddings, all_test_targets = [], []
    for batch_idx in tqdm(range(test_batches), desc="Computing Test Embeddings"):
        x_batch, y_batch = next(test_iterator)
        batch_embeddings = encoder_model(x_batch)
        all_test_embeddings.append(batch_embeddings.numpy())
        all_test_targets.append(y_batch.numpy())
    test_embeddings = np.concatenate(all_test_embeddings, axis=0)
    test_targets = np.concatenate(all_test_targets, axis=0)

    results_dict["test_dataset"]["embeddings_shape"] = str(test_embeddings.shape)
    results_dict["test_dataset"]["target_shape"] = str(test_targets.shape)
    np.savez_compressed(
        os.path.join(model_dir, "test_embeddings.npz"),
        embeddings=test_embeddings, 
        targets=test_targets,
    )
    del test_transformed, test_iterator, all_test_embeddings, all_test_targets, test_embeddings, test_targets

    with open(os.path.join(model_dir, "results.json"), 'w') as f:
        json.dump(results_dict, f, indent=4)

if __name__ == "__main__":
    main()
