import os
import numpy as np
import json
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys

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
CHECKPOINT_DIRECTORY = os.path.join(SCRIPT_DIR, "downstream_checkpoints")
RUN_ID = 1
# Model Hyperparameters
EMBEDDING_DIM = 64
BATCHSIZE = 128
EPOCHS = 20

os.makedirs(CHECKPOINT_DIRECTORY, exist_ok=True)


def main():
    train = np.load(os.path.join(SCRIPT_DIR, "train_embeddings.npz"))
    X_train_embed = train["embeddings"]
    y_train_scaled = train["targets"]

    val = np.load(os.path.join(SCRIPT_DIR, "val_embeddings.npz"))
    X_val_embed = val["embeddings"]
    y_val_scaled = val["targets"]

    test = np.load(os.path.join(SCRIPT_DIR, "test_embeddings.npz"))
    X_test_embed = test["embeddings"]
    y_test_scaled = test["targets"]
    
    with open(os.path.join(DATA_DIRECTORY, "y_scaler.pkl"), 'rb') as f:
        y_scaler = pickle.load(f)

    # Construct Neural Network
    input = tf.keras.layers.Input(shape=(EMBEDDING_DIM,))
    nn = tf.keras.Sequential([
        tf.keras.layers.Dense(32),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(16),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(8),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
    ])(input)
    output = tf.keras.layers.Dense(1)(nn)
    downstream_model = tf.keras.Model(inputs=input, outputs=output)

    downstream_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), loss="mse")
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(CHECKPOINT_DIRECTORY, f"backup_{RUN_ID}")
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=6,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIRECTORY, f"best_model_{RUN_ID}.keras"),
            monitor='val_loss',
            save_best_only=True
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(CHECKPOINT_DIRECTORY, f"log_{RUN_ID}.csv"),
            append=True,
        )
    ]

    downstream_model.fit(
        X_train_embed, y_train_scaled,
        validation_data=(X_val_embed, y_val_scaled),
        epochs=EPOCHS,
        batch_size=BATCHSIZE,
        callbacks=callbacks,
    )

    y_contrastive_scaled = downstream_model.predict(X_test_embed, verbose=1)
    
    y_contrastive = y_scaler.inverse_transform(y_contrastive_scaled).flatten()
    y_test = y_scaler.inverse_transform(y_test_scaled).flatten()
    
    downstream_train_results = calculate_metrics(y_test, y_contrastive, "Downtream_NN")

    metrics = {
        **downstream_train_results,
    }

    with open(os.path.join(SCRIPT_DIR, f"downstream_prediction_results_{RUN_ID}.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
if __name__ == "__main__":
    main()