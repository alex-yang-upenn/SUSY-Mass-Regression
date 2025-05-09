import json
import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import sys
import tensorflow as tf
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(ROOT_DIR)
from graph_embeddings import GraphEmbeddings
from loss_functions import SimCLRNTXentLoss
from simCLR_model import *
from transformation import *
from utils import *
from plotting import *


# General Parameters
DATA_DIRECTORY = os.path.join(ROOT_DIR, "processed_data")
CHECKPOINT_DIRECTORY = os.path.join(SCRIPT_DIR, "checkpoints")
RUN_ID = 1
# Model Hyperparameters
N_FEATURES = 6
EMBEDDING_DIM = 64
OUTPUT_DIM = 48
BATCHSIZE = 128

os.makedirs(CHECKPOINT_DIRECTORY, exist_ok=True)


def main():
    # Test running the model with view pairs
    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled = load_data(DATA_DIRECTORY)
    train_batches = int(len(X_train_scaled) // BATCHSIZE)
    val_batches = int(len(X_val_scaled) // BATCHSIZE)
    test_batches = int(len(X_test_scaled) // BATCHSIZE)
    
    train_transformed_pairs = create_transformed_pairs_dataset(X_train_scaled, y_train_scaled, BATCHSIZE, N_FEATURES)
    train_transformed = create_transformed_dataset(X_train_scaled, y_train_scaled, BATCHSIZE, N_FEATURES)
    val_transformed_pairs = create_transformed_pairs_dataset(X_val_scaled, y_val_scaled, BATCHSIZE, N_FEATURES)
    val_transformed = create_transformed_dataset(X_val_scaled, y_val_scaled, BATCHSIZE, N_FEATURES)
    test_transformed_pairs = create_transformed_pairs_dataset(X_test_scaled, y_test_scaled, BATCHSIZE, N_FEATURES)
    test_transformed = create_transformed_dataset(X_test_scaled, y_test_scaled, BATCHSIZE, N_FEATURES)

    # loaded_model = tf.keras.models.load_model(
    #     os.path.join(CHECKPOINT_DIRECTORY, f"best_model_{RUN_ID}.keras"),
    #     custom_objects={"SimCLRNTXentLoss": SimCLRNTXentLoss, "GraphEmbeddings": GraphEmbeddings, "SimCLRModel": SimCLRModel}
    # )
    loaded_model = create_simclr_model(
        n_features=N_FEATURES,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM
    )
    for inputs, _ in train_transformed_pairs.take(1):
        _ = loaded_model(inputs)
        break
    loaded_model.load_weights(os.path.join(CHECKPOINT_DIRECTORY, f"best_model_{RUN_ID}.keras"))

    # Evaluate with siamese model
    train_loss = loaded_model.evaluate(train_transformed_pairs, steps=train_batches, verbose=1)
    results_dict = {
        "training_dataset_contrastive_loss": train_loss
    }
    val_loss = loaded_model.evaluate(val_transformed_pairs, steps=val_batches, verbose=1)
    results_dict["val_dataset_contrastive_loss"] = val_loss
    test_loss = loaded_model.evaluate(test_transformed_pairs, steps=test_batches, verbose=1)
    results_dict["test_dataset_contrastive_loss"] = test_loss

    # Run just the encoder to compute and store embeddings
    encoder_model = loaded_model.encoder
    encoder_model.save(os.path.join(SCRIPT_DIR, f"best_model_encoder.keras"))

    all_train_embeddings = []
    all_train_labels = []

    iterator = iter(train_transformed)
    for batch_idx in tqdm(range(train_batches), desc="Encoding train batches"):
        x_batch, y_batch = next(iterator)
        batch_embeddings = encoder_model(x_batch)
        all_train_embeddings.append(batch_embeddings.numpy())
        all_train_labels.append(y_batch.numpy())

    embeddings = np.concatenate(all_train_embeddings, axis=0)
    targets = np.concatenate(all_train_labels, axis=0)

    results_dict["train_embedding_shape"] = str(embeddings.shape)
    results_dict["train_target_shape"] = str(targets.shape)

    np.savez_compressed(
        os.path.join(SCRIPT_DIR, "train_embeddings.npz"),
        embeddings=embeddings, 
        targets=targets,
    )

    del all_train_embeddings, all_train_labels, embeddings, targets

    all_val_embeddings = []
    all_val_labels = []

    iterator = iter(val_transformed)
    for batch_idx in tqdm(range(val_batches), desc="Encoding val batches"):
        x_batch, y_batch = next(iterator)
        batch_embeddings = encoder_model(x_batch)
        all_val_embeddings.append(batch_embeddings.numpy())
        all_val_labels.append(y_batch.numpy())

    embeddings = np.concatenate(all_val_embeddings, axis=0)
    targets = np.concatenate(all_val_labels, axis=0)

    results_dict["val_embedding_shape"] = str(embeddings.shape)
    results_dict["val_target_shape"] = str(targets.shape)

    np.savez_compressed(
        os.path.join(SCRIPT_DIR, "val_embeddings.npz"),
        embeddings=embeddings, 
        targets=targets,
    )

    del all_val_embeddings, all_val_labels, embeddings, targets

    all_test_embeddings = []
    all_test_labels = []

    for x_batch, y_batch in tqdm(test_transformed, desc="Encoding test batches"):
        batch_embeddings = encoder_model(x_batch)
        all_test_embeddings.append(batch_embeddings.numpy())
        all_test_labels.append(y_batch.numpy())

    embeddings = np.concatenate(all_test_embeddings, axis=0)
    targets = np.concatenate(all_test_labels, axis=0)

    results_dict["test_embedding_shape"] = str(embeddings.shape)
    results_dict["test_target_shape"] = str(targets.shape)

    np.savez_compressed(
        os.path.join(SCRIPT_DIR, "test_embeddings.npz"),
        embeddings=embeddings, 
        targets=targets,
    )

    with open(os.path.join(SCRIPT_DIR, f"siamese_encoder_results_{RUN_ID}.json"), 'w') as f:
        json.dump(results_dict, f, indent=4)

if __name__ == "__main__":
    main()