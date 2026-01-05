"""TensorFlow/Keras callbacks for model training.

This module provides pre-configured callback sets for different training scenarios:
- Standard callbacks with early stopping and learning rate reduction
- No-stop callbacks for fixed-epoch training with LR scheduling
- Finetuning callbacks for progressive encoder freezing

Usage:
    from callbacks import get_standard_callbacks, get_no_stop_callbacks, get_finetuning_callbacks
"""

import os

import tensorflow as tf

from downstream_model import FinetuningCallback


def get_standard_callbacks(
    directory,
    early_stopping_patience=6,
    reduce_lr_patience=2,
):
    """Get standard training callbacks with early stopping.

    Returns a list of callbacks including backup/restore, early stopping based on
    validation loss, learning rate reduction on plateau, model checkpointing, and
    CSV logging of training metrics.

    Args:
        directory: Str, directory to save model checkpoints, backups, and logs.
        early_stopping_patience: Int, number of epochs with no improvement after
            which training will be stopped. Default 6.
        reduce_lr_patience: Int, number of epochs with no improvement after which
            learning rate will be reduced. Default 2.

    Returns:
        list: List of tf.keras.callbacks for model training.

    Example:
        >>> callbacks = get_standard_callbacks("models/gnn_baseline")
        >>> model.fit(X, y, callbacks=callbacks)
    """
    return [
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(directory, "backup"),
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(directory, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(directory, "training_logs.csv"),
            append=True,
        ),
    ]


def get_no_stop_callbacks(directory, learning_rate_decay=0.95):
    """Get callbacks for fixed-epoch training without early stopping.

    Returns callbacks including backup/restore, exponential learning rate decay,
    model checkpointing (best and last), and CSV logging. Suitable for contrastive
    pretraining where you want to train for a fixed number of epochs.

    Args:
        directory: Str, directory to save model checkpoints, backups, and logs.
        learning_rate_decay: Float, multiplicative factor for learning rate decay
            per epoch. Default 0.95.

    Returns:
        list: List of tf.keras.callbacks for model training.

    Example:
        >>> callbacks = get_no_stop_callbacks("models/siamese", learning_rate_decay=0.9)
        >>> model.fit(X, y, epochs=50, callbacks=callbacks)
    """
    return [
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(directory, "backup"),
        ),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * learning_rate_decay,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(directory, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(directory, "last_model.keras"),
            save_best_only=False,
            save_weights_only=False,
            save_freq="epoch",
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(directory, "training_logs.csv"),
            append=True,
        ),
    ]


def get_finetuning_callbacks(
    directory,
    freeze_epoch=3,
    low_lr=5e-6,
    normal_lr=1e-4,
    lr_decay_factor=0.90,
):
    """Get callbacks for progressive encoder freezing during finetuning.

    Returns callbacks including backup/restore, a custom FinetuningCallback that
    freezes the encoder after a specified epoch and adjusts learning rate, model
    checkpointing (best and last), and CSV logging. Used for finetuning pretrained
    models where you first train the downstream head, then freeze the encoder and
    continue training.

    Args:
        directory: Str, directory to save model checkpoints, backups, and logs.
        freeze_epoch: Int, epoch at which to freeze the encoder weights. Default 3.
        low_lr: Float, learning rate to use after freezing encoder. Default 5e-6.
        normal_lr: Float, learning rate to use before freezing. Default 1e-4.
        lr_decay_factor: Float, multiplicative factor for LR decay per epoch. Default 0.90.

    Returns:
        list: List of tf.keras.callbacks for model training.

    Example:
        >>> callbacks = get_finetuning_callbacks("models/siamese_finetune", freeze_epoch=3)
        >>> model.fit(X, y, epochs=25, callbacks=callbacks)
    """
    return [
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(directory, "backup"),
        ),
        FinetuningCallback(
            freeze_epoch=freeze_epoch,
            low_lr=low_lr,
            normal_lr=normal_lr,
            lr_decay_factor=lr_decay_factor,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(directory, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(directory, "last_model.keras"),
            save_best_only=False,
            save_weights_only=False,
            save_freq="epoch",
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(directory, "training_logs.csv"),
            append=True,
        ),
    ]
