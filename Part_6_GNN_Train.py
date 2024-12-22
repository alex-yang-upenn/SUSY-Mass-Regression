import keras_tuner as kt
import numpy as np
import tensorflow as tf

import datetime
import json
import os

from tf_utils import GraphEmbeddings


# General Parameters
DATA_DIRECTORY = "raw_data"
TRAINING_DIRECTORY = "GNN_Checkpoints"
RUN_ID = datetime.datetime.now().strftime('%m-%d_%H:%M')
# Model Parameters
N_FEATURES = 4
BATCHSIZE = 128
OUTPUT_SHAPE = 1
EPOCHS_PER_TRIAL = 15

os.makedirs(TRAINING_DIRECTORY, exist_ok=True)


def build_model():
    input = tf.keras.Input(shape=(N_FEATURES, None), dtype=tf.float32)

    # Reduce graph to vector embedding
    O_Bar = GraphEmbeddings()(input)
    
    # Trainable function phi_C to compute MET Eta from vector embeddings 
    dense1 = tf.keras.layers.Dense(64, activation="relu")(O_Bar)
    norm1 = tf.keras.layers.BatchNormalization()(dense1)
    dense2 = tf.keras.layers.Dense(32, activation="relu")(norm1)
    dense3 = tf.keras.layers.Dense(10, activation="relu")(dense2)
    output = tf.keras.layers.Dense(1)(dense3)
    
    # Create and compile model
    model = tf.keras.Model(inputs=input, outputs=output)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    return model

def main():
    # Load in data
    train = np.load(os.path.join(DATA_DIRECTORY, "train.npz"))
    val = np.load(os.path.join(DATA_DIRECTORY, "val.npz"))
    test = np.load(os.path.join(DATA_DIRECTORY, "test.npz"))
    X_train, y_train = train['X'], train['y_eta']
    X_val, y_val = val['X'], val['y_eta']
    X_test, y_test = test['X'], test['y_eta']
    
    def reformat_data(X):
        X_graphical = np.zeros((len(X), 16), dtype=np.float32)
        X_graphical[:, :12] = X[:, :12]
        X_graphical[:, 14] = X[:, 13]
        X_graphical = X_graphical.reshape(len(X), 4, 4)
        return X_graphical
    
    # Complete MET to a full "particle" with eta = 0 and mass = 0
    X_graphical_train = reformat_data(X_train)
    X_graphical_val = reformat_data(X_val)
    X_graphical_test = reformat_data(X_test)

    model = build_model()

    checkpoint_path = os.path.join(TRAINING_DIRECTORY, f"checkpoint_{RUN_ID}.keras")
    best_model_path = os.path.join(TRAINING_DIRECTORY, f"best_model_{RUN_ID}.keras")
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model = tf.keras.models.load_model(checkpoint_path, custom_objects={"GraphEmbeddings": GraphEmbeddings})

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=False,
            save_best_only=False,
            save_freq="epoch"
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(TRAINING_DIRECTORY, f"best_model_log_{RUN_ID}.csv")
        )
    ]
    
    model.fit(
        X_graphical_train,
        y_train,
        validation_data=(X_graphical_val, y_val),
        epochs=EPOCHS_PER_TRIAL,
        batch_size=BATCHSIZE,
        callbacks=callbacks,
    )

    # Evaluate on test set
    test_results = model.evaluate(X_graphical_test, y_test, verbose=1)

    # Save results
    results_dict = {
        "best_model": checkpoint_path,
        "best_model_metrics": {
            "test_mse": float(test_results[0]),
            "test_mae": float(test_results[1]),
            "test_mape": float(test_results[2]),
        }
    }
    with open(os.path.join(TRAINING_DIRECTORY, f"tuning_results_{RUN_ID}.json"), 'w') as f:
        json.dump(results_dict, f, indent=4)

if __name__ == "__main__":
    main()
