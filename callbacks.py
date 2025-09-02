"""
Module Name: callbacks

Description:
    TensorFlow callback functions for training models. These callbacks are parameterized
    and can be used across different model training scripts.

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
    """
    Standard callbacks with early stopping and learning rate reduction.

    Args:
        directory (str): Directory to save model and logs
        early_stopping_patience (int): Patience for early stopping
        reduce_lr_patience (int): Patience for learning rate reduction

    Returns:
        list: List of TensorFlow callbacks
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


def get_no_stop_callbacks(directory, learning_rate_decay=0.9):
    """
    Callbacks without early stopping, using learning rate scheduling.

    Args:
        directory (str): Directory to save model and logs
        learning_rate_decay (float): Learning rate decay factor per epoch

    Returns:
        list: List of TensorFlow callbacks
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
    """
    Callbacks for finetuning with encoder freezing.

    Args:
        directory (str): Directory to save model and logs
        freeze_epoch (int): Epoch at which to freeze encoder
        low_lr (float): Learning rate after freezing
        normal_lr (float): Learning rate before freezing
        lr_decay_factor (float): Learning rate decay factor

    Returns:
        list: List of TensorFlow callbacks
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
