import json
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(ROOT_DIR)
from graph_embeddings import GraphEmbeddings
from transformation import ViewTransformedGenerator
from utils import *

DATA_DIRECTORY = os.path.join(ROOT_DIR, "processed_data")
MODEL_NAME = "best_model_1.keras"
N_FEATURES = 6
BATCHSIZE = 128


def create_transformed_dataset(X, y):
    # Create TensorFlow datasets
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(BATCHSIZE)

    view_transformed_generator = ViewTransformedGenerator(dataset)

    output_signature = (
        tf.TensorSpec(shape=(None, N_FEATURES, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    )

    transformed_dataset = tf.data.Dataset.from_generator(
        view_transformed_generator.generate,
        output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE)

    return transformed_dataset


def main():
    # Predictions given by model
    _, _, _, _, X, y = load_data(DATA_DIRECTORY)
    with open(os.path.join(DATA_DIRECTORY, "y_scaler.pkl"), 'rb') as f:
        y_scaler = pickle.load(f)

    test_size = len(X)
    test_batches = int(test_size // BATCHSIZE)

    test_transformed = create_transformed_dataset(X, y)
    
    model = tf.keras.models.load_model(os.path.join(SCRIPT_DIR, "checkpoints", MODEL_NAME))
    
    y_model_list = []
    y_true_list = []
    for features, labels in tqdm(test_transformed, desc="Processing batches", total=test_batches):
        y_model_batch_scaled = model.predict(features, verbose=0)
        
        y_model_batch = y_scaler.inverse_transform(y_model_batch_scaled)
        y_model_list.append(y_model_batch)
        y_true_batch = y_scaler.inverse_transform(labels.numpy())
        y_true_list.append(y_true_batch)
    y_model = np.concatenate(y_model_list).flatten()
    y_true = np.concatenate(y_true_list).flatten()

    # Predictions given by naive lorentz addition
    _, _, _, _, X_orig, y_orig = load_data_original(DATA_DIRECTORY)
    
    X_orig = X_orig.transpose(0, 2, 1)
    y_orig = y_orig.reshape(-1, 1)

    test_size = len(X_orig)
    test_batches = int(test_size // BATCHSIZE)

    test_orig_transformed = create_transformed_dataset(X_orig, y_orig)

    y_lorentz_list = []
    y_lorentz_true_list = []
    for features, labels in tqdm(test_orig_transformed, desc="Processing batches", total=test_batches):
        particles = features.numpy().transpose(0, 2, 1)
        particle_masses = np.zeros((particles.shape[0], particles.shape[1]))
        y = vectorized_lorentz_addition(particles, particle_masses)
        y_lorentz_list.append(y)
        y_lorentz_true_list.append(labels.numpy())
    y_lorentz = np.concatenate(y_lorentz_list).flatten()
    y_lorentz_true = np.concatenate(y_lorentz_true_list).flatten()

    model_metrics = calculate_metrics(y_true, y_model, "best_model_1.keras")
    lorentz_metrics = calculate_metrics(y_lorentz_true, y_lorentz, "lorentz")

    metrics = {
        **model_metrics,
        **lorentz_metrics
    }
    with open(os.path.join(SCRIPT_DIR, "transformed_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()