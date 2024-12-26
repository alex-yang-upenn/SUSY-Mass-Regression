import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras_tuner as kt
import numpy as np
import tensorflow as tf

import datetime
import json

from tf_utils import GraphEmbeddings
from utils import reformat_data


# General Parameters
DATA_DIRECTORY = "pre-processed_data"
TRAINING_DIRECTORY = "GNN_Checkpoints"
RUN_ID = "12-24_22:03" # datetime.datetime.now().strftime('%m-%d_%H:%M')
# Model Parameters
N_FEATURES = 4
BATCHSIZE = 128
OUTPUT_SHAPE = 1
MAX_TRIALS = 30
EPOCHS_PER_TRIAL = 15

os.makedirs(TRAINING_DIRECTORY, exist_ok=True)


def build_model(hp):
    input = tf.keras.Input(shape=(N_FEATURES, None), dtype=tf.float32)

    # Reduce graph to vector embedding
    hp_fr_dense1 = hp.Int('fr_dense1', min_value=32, max_value=128, step=32)
    hp_fr_dense2 = hp.Int('fr_dense2', min_value=16, max_value=64, step=16)
    hp_fr_dense3 = hp.Int('fr_dense3', min_value=8, max_value=32, step=8)
    hp_fo_dense1 = hp.Int('fo_dense1', min_value=32, max_value=128, step=32)
    hp_fo_dense2 = hp.Int('fo_dense2', min_value=16, max_value=64, step=16)
    hp_fo_dense3 = hp.Int('fo_dense3', min_value=8, max_value=32, step=8)
    O_Bar = GraphEmbeddings(
        f_r_units=(hp_fr_dense1, hp_fr_dense2, hp_fr_dense3),
        f_o_units=(hp_fo_dense1, hp_fo_dense2, hp_fo_dense3)
    )(input)
    
    # Trainable function phi_C to compute MET Eta from vector embeddings 
    hp_phi_C_dense1 = hp.Int('phi_C_dense1', min_value=32, max_value=128, step=32)
    dense1 = tf.keras.layers.Dense(
        units=hp_phi_C_dense1,
        activation="relu"
    )(O_Bar)
    norm1 = tf.keras.layers.BatchNormalization()(dense1)

    hp_phi_C_dense2 = hp.Int('phi_C_dense2', min_value=16, max_value=64, step=16)
    dense2 = tf.keras.layers.Dense(
        units=hp_phi_C_dense2,
        activation="relu"
    )(norm1)
    norm2 = tf.keras.layers.BatchNormalization()(dense2)

    hp_phi_C_dense3 = hp.Int('phi_C_dense3', min_value=8, max_value=32, step=8)
    dense3 = tf.keras.layers.Dense(
        units=hp_phi_C_dense3,
        activation="relu"
    )(norm2)
    norm3 = tf.keras.layers.BatchNormalization()(dense3)

    output = tf.keras.layers.Dense(1)(norm3)
    
    # Create and compile model
    model = tf.keras.Model(inputs=input, outputs=output)
    
    hp_learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    return model

def main():
    print("GPUs Available: ", tf.config.list_physical_devices("GPU"))
    
    # Load in data
    train = np.load(os.path.join(DATA_DIRECTORY, "train.npz"))
    val = np.load(os.path.join(DATA_DIRECTORY, "val.npz"))
    test = np.load(os.path.join(DATA_DIRECTORY, "test.npz"))
    X_train, y_train = train['X'], train['y_eta']
    X_val, y_val = val['X'], val['y_eta']
    X_test, y_test = test['X'], test['y_eta']
    
    # Complete MET to a full "particle" with eta = 0 and mass = 0
    X_graphical_train = reformat_data(X_train)
    X_graphical_val = reformat_data(X_val)
    X_graphical_test = reformat_data(X_test)

    # Define the tuner
    tuner = kt.BayesianOptimization(
        build_model,
        objective='val_loss',
        max_trials=MAX_TRIALS,
        directory=os.path.join(TRAINING_DIRECTORY, f'tuning_{RUN_ID}'),
        project_name='eta_prediction'
    )

    # Define callbacks for each trial
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
        )
    ]

    search_space_summary = tuner.search_space_summary(extended=True)

    # Search for best hyperparameters
    tuner.search(
        X_graphical_train, y_train,
        validation_data=(X_graphical_val, y_val),
        epochs=EPOCHS_PER_TRIAL,
        batch_size=BATCHSIZE,
        callbacks=callbacks
    )

    results_summary = tuner.results_summary(30)

    # Get best hyperparameters, build and train best model
    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = build_model(best_hp)

    best_checkpoint_path = os.path.join(TRAINING_DIRECTORY, f"best_model_{RUN_ID}.keras")
    best_model_callbacks = callbacks + [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=best_checkpoint_path,
                monitor='val_loss',
                save_best_only=True
            ),
            tf.keras.callbacks.CSVLogger(
                os.path.join(TRAINING_DIRECTORY, f"best_model_log_{RUN_ID}.csv")
            )
    ]

    best_model.fit(
        X_graphical_train, y_train,
        validation_data=(X_graphical_val, y_val),
        epochs=EPOCHS_PER_TRIAL,
        batch_size=BATCHSIZE,
        callbacks=best_model_callbacks
    )

    # Evaluate on test set
    test_results = best_model.evaluate(X_graphical_test, y_test, verbose=1)

    results_dict = {
        "best_model": best_checkpoint_path,
        "best_model_metrics": {
            "test_mse": float(test_results[0]),
            "test_mae": float(test_results[1]),
            "test_mape": float(test_results[2]),
        },
        "best_hyperparameters": best_hp.values,
        "search_space_summary": search_space_summary,
        "results_summary": results_summary
    }

    # Save results
    with open(os.path.join(TRAINING_DIRECTORY, f"tuning_results_{RUN_ID}.json"), 'w') as f:
        json.dump(results_dict, f, indent=4)

if __name__ == "__main__":
    main()
