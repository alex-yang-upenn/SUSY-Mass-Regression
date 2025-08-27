import os
import sys
import tensorflow as tf

from downstream_model import FinetuningCallback

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIRECTORY = os.path.join(ROOT_DIR, "raw_data_set2")
PROCESSED_DATA_DIRECTORY = os.path.join(ROOT_DIR, "processed_data_set2")

EVAL_DATA_FILES = [
    "test_bA_bgX_bgqqqqY_bgqqqqqqqqlv_A2500_X1000_Y500.npz",
    "test_bA_bgX_bgqqqqY_bgqqqqqqqqlv_A3300_X1400_Y550.npz",
    "test_bA_bgX_bgqqqqY_bgqqqqqqqqlv_A4100_X1800_Y600.npz",
    "test_bA_bgX_bgqqqqY_bgqqqqqqqqlv_A5300_X2400_Y800.npz",
    "test_bA_bgX_bgqqqqY_bgqqqqqqqqlv_A6500_X3000_Y1000.npz",
]


DECAY_CHAIN = ["P1", "P2"]
N_PARTICLES = 12
N_FEATURES = 7
MET_IDS = set([12]) 
LEPTON_IDS = set([11])
GLUON_IDS = set([21])
TRAIN_TEST_SPLIT = 0.2
SCALABLE_FEATURES = [0, 1, 2]


GNN_BASELINE_F_R_LAYER_SIZES = (96, 64, 32)
GNN_BASELINE_F_O_LAYER_SIZES = (128, 96, 64)
GNN_BASELINE_PHI_C_LAYER_SIZES = (16, 8, 4)
GNN_BASELINE_LEARNING_RATE=5e-4
GNN_BASELINE_EARLY_STOPPING_PATIENCE=6
GNN_BASELINE_REDUCE_LR_PATIENCE=2

GNN_TRANSFORMED_LEARNING_RATE=5e-4
GNN_TRANSFORMED_LEARNING_RATE_DECAY=0.9

SIAMESE_PHI_C_LAYER_SIZES = (32,)
SIAMESE_PROJ_HEAD_LAYER_SIZES = (48, 64)
SIAMESE_LEARNING_RATE=1e-4

DOWNSTREAM_FREEZE_EPOCH=3
DOWNSTREAM_LEARNING_RATE=5e-6
DOWNSTREAM_FROZEN_LEARNING_RATE=1e-4
DOWNSTREAM_LR_DECAY=0.90

DOWNSTREAM_NO_FINETUNE_LEARNING_RATE=1e-4

SIAMESE_DOWNSTREAM_LAYER_SIZES = (8, 4)

BATCHSIZE = 256
EPOCHS = 15
SIAMESE_EPOCHS = 20
SIMCLR_LOSS_TEMP = 0.1
RUN_ID = 1

def STANDARD_CALLBACKS(
        directory,
        early_stopping_patience=GNN_BASELINE_EARLY_STOPPING_PATIENCE,
        reduce_lr_patience=GNN_BASELINE_REDUCE_LR_PATIENCE,
    ):
    return [
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(directory, "backup"),
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(directory, "best_model.keras"),
            monitor='val_loss',
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(directory, "training_logs.csv"),
            append=True,
        )
    ]


def NO_STOP_CALLBACKS(directory):
    return [
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(directory, "backup"),
        ),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * GNN_TRANSFORMED_LEARNING_RATE_DECAY,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(directory, "best_model.keras"),
            monitor='val_loss',
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
        )
    ]


def FINETUNING_CALLBACKS(directory):
    return [
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(directory, "backup"),
        ),
        FinetuningCallback(
            freeze_epoch=DOWNSTREAM_FREEZE_EPOCH,
            low_lr=DOWNSTREAM_LEARNING_RATE,
            normal_lr=DOWNSTREAM_FROZEN_LEARNING_RATE,
            lr_decay_factor=DOWNSTREAM_LR_DECAY,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(directory, "best_model.keras"),
            monitor='val_loss',
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
        )
    ]
