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
from simCLR_model import *
from transformation import ViewPairsGenerator, create_transformed_pairs_dataset
from utils import *


# General Parameters
DATA_DIRECTORY = os.path.join(ROOT_DIR, "processed_data")
CHECKPOINT_DIRECTORY = os.path.join(SCRIPT_DIR, "checkpoints_test")
RUN_ID = 1 # datetime.datetime.now().strftime('%m-%d_%H:%M')
# Model Hyperparameters
N_FEATURES = 6
EMBEDDING_DIM = 64
OUTPUT_DIM = 48
BATCHSIZE = 128
EPOCHS = 20

os.makedirs(CHECKPOINT_DIRECTORY, exist_ok=True)


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(DATA_DIRECTORY)
    train_size = len(X_train)
    val_size = len(X_val)
    test_size = len(X_test)
    train_batches = int(train_size // BATCHSIZE)
    val_batches = int(val_size / BATCHSIZE)
    test_batches = int(test_size // BATCHSIZE)

    train_pairs = create_transformed_pairs_dataset(X_train, y_train, BATCHSIZE, N_FEATURES)

    val_pairs = create_transformed_pairs_dataset(X_val, y_val, BATCHSIZE, N_FEATURES)

    test_pairs = create_transformed_pairs_dataset(X_test, y_test, BATCHSIZE, N_FEATURES)

    simclr_model = create_simclr_model(
        n_features=N_FEATURES,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM
    )

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
            append=True
        )
    ]

    # Call the model once to build it
    for inputs, _ in train_pairs.take(1):
        _ = simclr_model(inputs)
        break

    simclr_model.fit(
        train_pairs,
        validation_data=val_pairs,
        epochs=EPOCHS,
        steps_per_epoch=train_batches,
        validation_steps=val_batches,
        callbacks=callbacks
    )

    test_loss = simclr_model.evaluate(test_pairs, steps=test_batches, verbose=1)

    results_dict = {
        "contrastive_loss_test_dataset": test_loss
    }

    with open(os.path.join(CHECKPOINT_DIRECTORY, f"results_{RUN_ID}.json"), 'w') as f:
        json.dump(results_dict, f, indent=4)

if __name__ == "__main__":
    main()