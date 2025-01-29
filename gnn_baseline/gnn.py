import datetime
import json
import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import sys
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from tf_utils import GraphEmbeddings


# General Parameters
DATA_DIRECTORY = os.path.join(os.path.dirname(SCRIPT_DIR), "processed_data")
RUN_ID = datetime.datetime.now().strftime('%m-%d_%H:%M')
# Model Hyperparameters
N_FEATURES = 6
BATCHSIZE = 128
EPOCHS = 20

os.makedirs(os.path.join(SCRIPT_DIR, "checkpoints"), exist_ok=True)


def normalize_data(train, scalable_particle_features):
    scalers = []
    scaled_train = train.copy()
    for i in scalable_particle_features:
        values = train[:, :, i]
        values_flat = values.reshape(-1, 1)

        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(values_flat)

        scaled_train[:, :, i] = scaled_values.reshape(values.shape)
        scalers.append(scaler)

    return scaled_train, scalers

def scale_data(data, scalers, scalable_particle_features):
    scaled_data = data.copy()
    for scaler, idx in zip(scalers, scalable_particle_features):
        values = data[:, :, idx]
        values_flat = values.reshape(-1, 1)

        scaled_values = scaler.transform(values_flat)

        scaled_data[:, :, idx] = scaled_values.reshape(values.shape)
    
    return scaled_data

def build_model():
    input = tf.keras.Input(shape=(N_FEATURES, None), dtype=tf.float32)

    # Reduce graph to vector embeddings
    O_Bar = GraphEmbeddings(
        f_r_units=(96, 64, 32),
        f_o_units=(128, 64, 8)
    )(input)
    
    # Trainable function phi_C to compute MET Eta from vector embeddings
    dense1 = tf.keras.layers.Dense(units=128)(O_Bar)
    norm1 = tf.keras.layers.BatchNormalization()(dense1)
    dense1_out = tf.keras.layers.ReLU()(norm1)
    

    dense2 = tf.keras.layers.Dense(units=32)(dense1_out)
    norm2 = tf.keras.layers.BatchNormalization()(dense2)
    dense2_out = tf.keras.layers.ReLU()(norm2)

    dense3 = tf.keras.layers.Dense(units=24)(dense2_out)
    norm3 = tf.keras.layers.BatchNormalization()(dense3)
    dense3_out = tf.keras.layers.ReLU()(norm3)

    output = tf.keras.layers.Dense(1)(dense3_out)
    
    # Create and compile model
    model = tf.keras.Model(inputs=input, outputs=output)
    
    learning_rate = 5e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    return model

def main():
    print("GPUs Available: ", tf.config.list_physical_devices("GPU"))
    
    X_trains, y_trains = [], []
    X_vals, y_vals = [], []
    X_tests, y_tests = [], []

    for name in os.listdir(os.path.join(DATA_DIRECTORY, "test")):
        if name[-4:] != ".npz":
            continue
        
        train = np.load(os.path.join(DATA_DIRECTORY, "train", name))
        val = np.load(os.path.join(DATA_DIRECTORY, "val", name))
        test = np.load(os.path.join(DATA_DIRECTORY, "test", name))

        X_trains.append(train['X'])
        y_trains.append(train['y'])
        X_vals.append(val['X'])
        y_vals.append(val['y'])
        X_tests.append(test['X'])
        y_tests.append(test['y'])

    X_train = np.concatenate(X_trains, axis=0)
    y_train = np.concatenate(y_trains, axis=0)
    X_val = np.concatenate(X_vals, axis=0)
    y_val = np.concatenate(y_vals, axis=0)
    X_test = np.concatenate(X_tests, axis=0)
    y_test = np.concatenate(y_tests, axis=0)
    
    X_train_scaled, scalers = normalize_data(X_train, [0, 1, 2])
    X_val_scaled = scale_data(X_val, scalers, [0, 1, 2])
    X_test_scaled = scale_data(X_test, scalers, [0, 1, 2])
    
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

    del X_train, X_val, X_test, y_train, y_val, y_test
    del X_trains, X_vals, X_tests, y_trains, y_vals, y_tests

    for i, s in enumerate(scalers):
        with open(os.path.join(DATA_DIRECTORY, f"x_scaler_{i}.pkl"), 'wb') as f:
            pickle.dump(s, f)
    
    with open(os.path.join(DATA_DIRECTORY, "y_scaler.pkl"), 'wb') as f:
        pickle.dump(y_scaler, f)

    del scalers, y_scaler

    # Change from (batchsize, num particles, num features) to (batchsize, num features, num particles)
    X_train_scaled = X_train_scaled.transpose(0, 2, 1)
    X_val_scaled = X_val_scaled.transpose(0, 2, 1)
    X_test_scaled = X_test_scaled.transpose(0, 2, 1)

    # Define callbacks for each trial
    best_checkpoint_path = os.path.join(SCRIPT_DIR, f"checkpoints/best_model_{RUN_ID}.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_checkpoint_path,
            monitor='val_loss',
            save_best_only=True
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(SCRIPT_DIR, f"checkpoints/log_{RUN_ID}.csv")
        )
    ]

    model = build_model()

    model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=EPOCHS,
        batch_size=BATCHSIZE,
        callbacks=callbacks
    )

    test_results = model.evaluate(X_test_scaled, y_test_scaled, verbose=1)

    results_dict = {
        "model": best_checkpoint_path,
        "best_model_metrics": {
            "test_mse": float(test_results[0]),
            "test_mae": float(test_results[1]),
            "test_mape": float(test_results[2]),
        }
    }

    with open(os.path.join(SCRIPT_DIR, f"results_{RUN_ID}.json"), 'w') as f:
        json.dump(results_dict, f, indent=4)


if __name__ == "__main__":
    main()