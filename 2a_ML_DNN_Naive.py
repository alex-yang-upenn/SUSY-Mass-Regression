import tensorflow as tf
import numpy as np

import datetime
import json
import os


# Parameters
DATA_DIRECTORY = "Preprocessed_Data"
TRAINING_DIRECTORY = "DNN_checkpoints"
RUN_ID = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
INPUT_SHAPE = 14
OUTPUT_SHAPE = 1
BATCHSIZE = 128

os.makedirs("DNN_checkpoints", exist_ok=True)


def main():
    train = np.load(os.path.join(DATA_DIRECTORY, "train.npz"))
    val = np.load(os.path.join(DATA_DIRECTORY, "val.npz"))
    X_train, y_train = train['X'], train['y_eta']
    X_val, y_val = val['X'], val['y_eta']

    # Defining the model
    inputs = tf.keras.layers.Input(shape=(INPUT_SHAPE,))

    x = tf.keras.layers.Dense(64, kernel_initializer="he_normal")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(32, kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    outputs = tf.keras.layers.Dense(OUTPUT_SHAPE, kernel_initializer="he_normal")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae", "mape"]
    )

    best_checkpoint_path = os.path.join(TRAINING_DIRECTORY, f"best_model_{RUN_ID}.keras")
    last_checkpoint_path = os.path.join(TRAINING_DIRECTORY, f"last_model_{RUN_ID}.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=last_checkpoint_path,
            save_best_only=False,
            verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(TRAINING_DIRECTORY, f"log_{RUN_ID}.csv")
        )
    ]
        
    # Train model
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=BATCHSIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Clean-up
    del train, val, X_train, X_val, y_train, y_val
    
    test = np.load(os.path.join(DATA_DIRECTORY, "test.npz"))
    X_test, y_test = test['X'], test['y_eta']
    test_results = model.evaluate(X_test, y_test, verbose=1)
    
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