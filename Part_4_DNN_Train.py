"""
Module Name: DNN

Description:
    This module treats the MET as a single particle, and trains a DNN to predict its Eta value.
    Then, it sums the Lorentz Vectors of all the visible particles at the end of the decay chain
    as well as the predicted MET for mass regression.

    DNN training is hyperparameter-tuned with Bayesian Optimization.

Usage:
Author:
Date:
License:
"""

import keras_tuner as kt
import numpy as np
import tensorflow as tf

import datetime
import json
import os


# General parameters
DATA_DIRECTORY = "pre-processed_data"
TRAINING_DIRECTORY = "DNN_Checkpoints"
RUN_ID = "12-01_13:32"  # datetime.datetime.now().strftime('%m-%d_%H:%M')
# Model parameters
INPUT_SHAPE = 14
OUTPUT_SHAPE = 1
BATCHSIZE = 32
# Keras Tuner parameters
MAX_TRIALS = 30         
EPOCHS_PER_TRIAL = 15

os.makedirs(TRAINING_DIRECTORY, exist_ok=True)


def build_model(hp):
    """
    Define the model architecture with the following tunable hyperparameters:
        - Number of units in each layer
        - Dropout rates
        - Kernel initializers
        - Learning rate
        - Optional third layer
    """

    inputs = tf.keras.layers.Input(shape=(INPUT_SHAPE,))
    
    # First layer
    x = tf.keras.layers.Dense(
        units=hp.Int('units_1', min_value=32, max_value=256, step=32),
        kernel_initializer=hp.Choice('kernel_init_1', ['he_normal', 'glorot_uniform'])
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(
        hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)
    )(x)
    
    # Second layer
    x = tf.keras.layers.Dense(
        units=hp.Int('units_2', min_value=16, max_value=128, step=16),
        kernel_initializer=hp.Choice('kernel_init_2', ['he_normal', 'glorot_uniform'])
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(
        hp.Float('dropout_2', min_value=0.0, max_value=0.3, step=0.1)
    )(x)
    
    # Optional third layer
    if hp.Boolean('use_third_layer'):
        x = tf.keras.layers.Dense(
            units=hp.Int('units_3', min_value=8, max_value=64, step=8),
            kernel_initializer=hp.Choice('kernel_init_3', ['he_normal', 'glorot_uniform'])
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(
            hp.Float('dropout_3', min_value=0.0, max_value=0.3, step=0.1)
        )(x)
    
    outputs = tf.keras.layers.Dense(OUTPUT_SHAPE)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model with tunable learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    return model


def main():
    train = np.load(os.path.join(DATA_DIRECTORY, "train.npz"))
    val = np.load(os.path.join(DATA_DIRECTORY, "val.npz"))
    test = np.load(os.path.join(DATA_DIRECTORY, "test.npz"))
    X_train, y_train = train['X'], train['y_eta']
    X_val, y_val = val['X'], val['y_eta']
    X_test, y_test = test['X'], test['y_eta']

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
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_PER_TRIAL,
        batchsize=BATCHSIZE,
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
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_PER_TRIAL,
        batchsize=BATCHSIZE,
        callbacks=best_model_callbacks
    )
    
    # Evaluate on test set
    test_results = best_model.evaluate(X_test, y_test, verbose=1)
    
    # Save results and best hyperparameters
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
    
    with open(os.path.join(TRAINING_DIRECTORY, f"tuning_results_{RUN_ID}.json"), 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    
if __name__ == "__main__":
    main()