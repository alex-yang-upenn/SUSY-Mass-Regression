import tensorflow as tf
import numpy as np

import datetime
import json
import os

from DataLoader import DataLoader


# Parameters
TRAIN_DATA_DIRECTORY = "Preprocessed_Data/Train"
TEST_DATA_DIRECTORY = "Preprocessed_Data/Test"
TRAINING_DIRECTORY = "DNN_checkpoints"
RUN_ID = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
INPUT_SHAPE = 14
OUTPUT_SHAPE = 1
BATCHSIZE = 32

os.makedirs("DNN_checkpoints", exist_ok=True)


def main():
    # Defining the model
    inputs = tf.keras.layers.Input(shape=(INPUT_SHAPE,))

    x = tf.keras.layers.Dense(64)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    outputs = tf.keras.layers.Dense(OUTPUT_SHAPE)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,
        weight_decay=1e-5,
        amsgrad=True,
    )

    train_data_generator = DataLoader(data_dir=TRAIN_DATA_DIRECTORY, batchsize=BATCHSIZE, shuffle=True)
    test_data_generator = DataLoader(data_dir=TEST_DATA_DIRECTORY, batchsize=BATCHSIZE, shuffle=True)

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae", "mape"]
    )

    best_checkpoint_path = os.path.join(TRAINING_DIRECTORY, f"best_model_{RUN_ID}.keras")
    last_checkpoint_path = os.path.join(TRAINING_DIRECTORY, f"last_model_{RUN_ID}.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_checkpoint_path,
            monitor="loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=last_checkpoint_path,
            save_best_only=False,
            verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=2,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(TRAINING_DIRECTORY, f"log_{RUN_ID}.csv")
        )
    ]
        
    # Train model
    model.fit(train_data_generator, epochs=10, callbacks=callbacks, verbose=1)

    # Evaluate model
    if os.path.exists(best_checkpoint_path):
        print(f"Loading best model from {best_checkpoint_path}")
        best_model = tf.keras.models.load_model(best_checkpoint_path)
    else:
        print("No checkpoint found, using current model state")
        best_model = model
    
    test_results = best_model.evaluate(test_data_generator, verbose=1)
    
    results_dict = {
        "model_checkpoint": best_checkpoint_path,
        "metrics": {
            "test_mse": float(test_results[0]),
            "test_mae": float(test_results[1]),
            "test_mape": float(test_results[2]),
        }
    }
    with open(os.path.join(TRAINING_DIRECTORY, f"eval_{RUN_ID}.json"), 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    
if __name__ == "__main__":
    main()