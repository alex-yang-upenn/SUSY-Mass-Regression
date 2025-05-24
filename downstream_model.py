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

from graph_embeddings import *
from loss_functions import *
from simCLR_model import *


class FinetunedNN(tf.keras.Model):
    def __init__(
            self,
            encoder_path,
            encoder_architecture_path,
            downstream_units,
            output_dim=1,
            trainable_encoder=True,
            load_from_path=True,
            **kwargs
        ):
        super().__init__(**kwargs)
        
        self.encoder_path = encoder_path
        self.encoder_architecture_path = encoder_architecture_path
        self.downstream_units = downstream_units
        self.output_dim = output_dim
        self.trainable_encoder = trainable_encoder
        
        # Only load the original encoder if instantiating fresh model
        if load_from_path:
            self.encoder = tf.keras.models.load_model(
                encoder_path,
                custom_objects={
                    "SimCLRNTXentLoss": SimCLRNTXentLoss,
                    "GraphEmbeddings": GraphEmbeddings,
                },
                compile=False,
            )
            self.set_encoder_trainable(self.trainable_encoder)

            with open(self.encoder_architecture_path, "w") as f:
                f.write(self.encoder.to_json())
        else:
            with open(self.encoder_architecture_path, "r") as f:
                self.encoder = tf.keras.models.model_from_json(
                    f.read(),
                    custom_objects={
                        "SimCLRNTXentLoss": SimCLRNTXentLoss,
                        "GraphEmbeddings": GraphEmbeddings,
                    }
                )

        # Create downstream layers
        self.f_R = tf.keras.Sequential([
            layer
            for units in self.downstream_units
            for layer in [
                tf.keras.layers.Dense(units),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
            ]
        ])
        
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

    def build(self, input_shape):
        if not hasattr(self, "encoder"):
            raise AttributeError("Encoder not loaded properly")

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
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            'encoder_path': self.encoder_path,
            'encoder_architecture_path': self.encoder_architecture_path,
            'downstream_units': self.downstream_units,
            'output_dim': self.output_dim,
            'trainable_encoder': self.trainable_encoder,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        config['load_from_path'] = False
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
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.freeze_epoch and self.model.trainable_encoder:
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: Freezing encoder weights and increasing learning rate to {self.normal_lr}")
            
            # Freeze the encoder
            if hasattr(self.model, 'set_encoder_trainable'):
                self.model.set_encoder_trainable(False)
            else:
                print("Warning: Model does not have set_encoder_trainable method")
            
            self.model.optimizer.learning_rate.assign(self.normal_lr)
            
        elif epoch < self.freeze_epoch:
            if epoch == 0:
                if hasattr(self.model, 'set_encoder_trainable'):
                    self.model.set_encoder_trainable(True)

                self.model.optimizer.learning_rate.assign(self.low_lr)

        elif epoch > self.freeze_epoch:
            current_lr = float(self.model.optimizer.learning_rate.numpy())
            new_lr = current_lr * self.lr_decay_factor
            self.model.optimizer.learning_rate.assign(new_lr)
    
    def on_epoch_end(self, epoch, logs=None):
        if self.verbose > 0:
            current_lr = float(self.model.optimizer.learning_rate.numpy())
            encoder_status = "trainable" if self.model.trainable_encoder else "frozen"
            print(f"Epoch {epoch + 1} completed - LR: {current_lr:.2e}, Encoder: {encoder_status}")