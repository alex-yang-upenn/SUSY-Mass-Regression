import datetime
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
from transformation import ViewPairsGenerator
from utils import *


# General Parameters
DATA_DIRECTORY = os.path.join(ROOT_DIR, "processed_data")
CHECKPOINT_DIRECTORY = os.path.join(SCRIPT_DIR, "checkpoints")
RUN_ID = 0 # datetime.datetime.now().strftime('%m-%d_%H:%M')
# Model Hyperparameters
N_FEATURES = 6
EMBEDDING_DIM = 64
BATCHSIZE = 128
EPOCHS = 20

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
        tf.TensorSpec(shape=(1,), dtype=tf.float32)
    )

    transformed_dataset = tf.data.Dataset.from_generator(
        view_pairs_generator.generate,
        output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE)

    return transformed_dataset


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(DATA_DIRECTORY)

    train_pairs = create_transformed_dataset(X_train, y_train)