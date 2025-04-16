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
from transformation import ViewPairsGenerator
from utils import *


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


def create_transformed_dataset(X, y):
    # Create TensorFlow datasets
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(BATCHSIZE)

    view_pairs_generator = ViewPairsGenerator(dataset)

    output_signature = (
        (
            tf.TensorSpec(shape=(None, N_FEATURES, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, N_FEATURES, None), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    )

    transformed_dataset = tf.data.Dataset.from_generator(
        view_pairs_generator.generate,
        output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE)

    return transformed_dataset


def main():
    _, _, _, _, X_test, y_test = load_data(DATA_DIRECTORY)
    test_size = len(X_test)
    test_batches = int(test_size // BATCHSIZE)

    test_pairs = create_transformed_dataset(X_test, y_test)

    # loaded_model = tf.keras.models.load_model(
    #     os.path.join(CHECKPOINT_DIRECTORY, f"best_model_{RUN_ID}.keras"),
    #     custom_objects={"SimCLRNTXentLoss": SimCLRNTXentLoss, "GraphEmbeddings": GraphEmbeddings, "SimCLRModel": SimCLRModel}
    # )

    loaded_model = create_simclr_model(
        n_features=N_FEATURES,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM
    )

    # Call the model once to build it
    for inputs, _ in test_pairs.take(1):
        _ = loaded_model(inputs)
        break

    loaded_model.load_weights(os.path.join(CHECKPOINT_DIRECTORY, f"best_model_{RUN_ID}.keras"))

    test_loss = loaded_model.evaluate(test_pairs, steps=test_batches, verbose=1)

    results_dict = {
        "contrastive_loss_test_dataset": test_loss
    }

    with open(os.path.join(CHECKPOINT_DIRECTORY, f"results_{RUN_ID}.json"), 'w') as f:
        json.dump(results_dict, f, indent=4)

if __name__ == "__main__":
    main()