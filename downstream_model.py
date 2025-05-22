"""
Module Name: downstream_model

Description:
    This module includes a custom Tensorflow model combining a pretrained encoder with 
    downstream layers. It also includes a custom callback that controls finetuning

Usage:
Author:
Date:
License:
"""
import tensorflow as tf
import os

from graph_embeddings import *
from loss_functions import *
from simCLR_model import *


class FinetunedNN(tf.keras.Model):
    def __init__(self, encoder_path, downstream_units, output_dim=1, trainable_encoder=True):
        """
        Constructor
        
        Args:
            encoder_path (str): Path to the saved encoder model
            downstream_units (tuple of int): List of units for each hidden layer in downstream model
            output_dim (int): Dimension of the model output. 1 is used for the current data/task.
            trainable_encoder (bool): Whether encoder weights should be trainable
        """
        super().__init__()
        
        self.encoder_path = encoder_path
        self.downstream_units = downstream_units
        self.trainable_encoder = trainable_encoder
        
        self._load_encoder()
        
        # Create the downstream model
        self.f_R = tf.keras.Sequential([
            layer
            for units in self.downstream_units
            for layer in [
                tf.keras.layers.Dense(units),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
            ]
        ])
        
        self.output_dim = output_dim
        self.output_layer = tf.keras.layers.Dense(self.output_dim)

    def set_encoder_trainable(self, trainable):
        """
        Set whether the encoder weights are trainable or frozen. Propagates through all layers of encoder.

        Args:
            trainable (bool): Mode 
        """
        self.trainable_encoder = trainable
        self.encoder.trainable = trainable
        for layer in self.encoder.layers:
            layer.trainable = trainable
    
    def _load_encoder(self):
        """
        Helper method to load the encoder with custom objects
        """
        self.encoder = tf.keras.models.load_model(
            self.encoder_path,
            custom_objects={
                "SimCLRNTXentLoss": SimCLRNTXentLoss,
                "GraphEmbeddings": GraphEmbeddings,
            },
            compile=False,
        )
        self.set_encoder_trainable(self.trainable_encoder)

    def build(self, input_shape):
        if not hasattr(self, 'encoder'):
            self._load_encoder()

        dummy_input = tf.keras.layers.Input(shape=input_shape[1:])
        embeddings = self.encoder(dummy_input)
        
        self.f_R.build(embeddings.shape)
        
        self.output_layer.build(tf.TensorShape([None, self.downstream_units[-1]]))
        
        super().build(input_shape)

    def call(self, inputs, training=None):
        embeddings = self.encoder(inputs, training=training)
        
        x = self.f_R(embeddings, training=training)
        
        outputs = self.output_layer(x, training=training)

        return outputs
    
    def compute_output_shape(self, input_shape):
        """Compute the output shape of the model"""
        return (input_shape[0], self.output_dim)
    
    def get_config(self):
        """Return model configuration for serialization"""
        config = super().get_config()
        config.update({
            'encoder_path': self.encoder_path,
            'downstream_units': self.downstream_units,
            'output_dim': self.output_dim,
            'trainable_encoder': self.trainable_encoder,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class FinetuningCallback(tf.keras.callbacks.Callback):
    def __init__(self, freeze_epoch, low_lr, normal_lr, lr_decay_factor=0.95, verbose=1):
        """
        Initialize the progressive finetuning callback.
        
        Args:
            freeze_epoch (int): Epoch at which to freeze encoder
            low_lr (float): Low learning rate for initial epochs
            normal_lr (float): Normal learning rate after freezing encoder
            lr_decay_factor (float): Learning rate decay factor after freezing encoder
            verbose (int): Verbosity level (0=silent, 1=progress messages)
        """
        super().__init__()
        self.freeze_epoch = freeze_epoch
        self.low_lr = low_lr
        self.normal_lr = normal_lr
        self.lr_decay_factor = lr_decay_factor
        self.verbose = verbose
        self.encoder_frozen = False
        
    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the beginning of each epoch to adjust learning rate and encoder trainability.
        """
        if epoch == self.freeze_epoch and not self.encoder_frozen:
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: Freezing encoder weights and increasing learning rate to {self.normal_lr}")
            
            # Freeze the encoder
            if hasattr(self.model, 'set_encoder_trainable'):
                self.model.set_encoder_trainable(False)
                self.encoder_frozen = True
            else:
                print("Warning: Model does not have set_encoder_trainable method")
            
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.normal_lr)
            
        elif epoch < self.freeze_epoch:
            if epoch == 0:
                if hasattr(self.model, 'set_encoder_trainable'):
                    self.model.set_encoder_trainable(True)

                tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.low_lr)

        elif epoch > self.freeze_epoch:
            current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            new_lr = current_lr * self.lr_decay_factor
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
    
    def on_epoch_end(self, epoch, logs=None):
        if self.verbose > 0:
            current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            encoder_status = "frozen" if self.encoder_frozen else "trainable"
            print(f"Epoch {epoch + 1} completed - LR: {current_lr:.2e}, Encoder: {encoder_status}")