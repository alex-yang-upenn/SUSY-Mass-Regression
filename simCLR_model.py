"""
Module Name: SimCLR_Model

Description:
    This module provides a custom Tensorflow Model for SimCLR contrastive learning. The SimCLRModel is
    internally composed of an encoder linked to a projection head. The encoder is a standard GNN
    followed by dense layers. The projection head is a simple model with only dense layers. To improve
    learning, the SimCLR loss function is applied to the output of the projection head. For downstream
    tasks however, the embeddings produced by the encoder are used.

Usage:
Author:
Date:
License:
"""
import tensorflow as tf

from graph_embeddings import *
from loss_functions import SimCLRNTXentLoss


class SimCLRModel(tf.keras.Model):
    def __init__(
            self,
            n_features,
            f_r_units, 
            f_o_units,
            phi_C_units,
            proj_units,
            **kwargs
    ):
        super().__init__(**kwargs)
        
        # Store configurations
        self.n_features = n_features
        self.f_r_units = f_r_units
        self.f_o_units = f_o_units
        self.phi_C_units = phi_C_units
        self.proj_units = proj_units

        # Create models
        self.encoder = self._create_encoder_model(
            self.n_features,
            self.f_r_units,
            self.f_o_units,
            self.phi_C_units
        )
        
        self.projection_head = self._create_projection_head(
            self.proj_units
        )
    
    def build(self, input_shape):        
        # Two inputs [view1, view2], each with shape (batch_size, n_features, None)
        self.encoder.build(input_shape[0])
        self.projection_head.build((None, self.phi_C_units[-1]))

        super().build(input_shape)

    def call(self, inputs):
        # Unpack the two views
        view1, view2 = inputs
        
        # Get embeddings for both views using the shared encoder.
        # Input dimension (n_features, number of particles)
        # Output dimension (embedding_dim,)
        z1 = self.encoder(view1)
        z2 = self.encoder(view2)
        
        # Apply projection head to both embeddings
        # Input dimension (embedding_dim,)
        # Output dimension (output_dim,)
        p1 = self.projection_head(z1)
        p2 = self.projection_head(z2)
        
        return tf.stack([p1, p2], axis=1)
    
    def _create_encoder_model(
            self,
            n_features,
            f_r_units, 
            f_o_units,
            phi_C_units
    ):
        """
        Helper function to build the encoder. 

        Args:
            n_features (int): The number of features per particle.
            f_r_units (tuple of int): Size of each layer of the f_r function in the GNN.
            f_o_units (tuple of int): Size of each layer of the f_o function in the GNN.
            phi_C_units (tuple of int): 
                Size of the dense layers following the GNN. The dimension of the embeddings
                produced is equal to phi_C_units[-1]
        """
        input = tf.keras.layers.Input(shape=(n_features, None), dtype=tf.float32)
        
        # Reduce graph to vector embeddings
        O_Bar = GraphEmbeddings(
            f_r_units=f_r_units,
            f_o_units=f_o_units
        )(input)
        
        # Trainable function phi_C to transform GNN output to final embeddings
        phi_C = tf.keras.Sequential([
            layer
            for units in phi_C_units
            for layer in [
                tf.keras.layers.Dense(units),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
            ]
        ])(O_Bar)

        return tf.keras.Model(inputs=input, outputs=phi_C)


    def _create_projection_head(
            self,
            proj_units,
    ):
        """
        Helper function to create Projection Head to be applied to embeddings. Shown in
        SimCLR paper to improve training.

        Args:
            proj_units (tuple of int): 
                Size of the dense layers. The SimCLR loss is applied to the output with
                dimensions equal to proj_units[-1]
        """
        proj_input = tf.keras.layers.Input(shape=(self.phi_C_units[-1],), dtype=tf.float32)
        
        proj_function = tf.keras.Sequential([
            layer
            for units in proj_units
            for layer in [
                tf.keras.layers.Dense(units),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
            ]
        ])(proj_input)
        
        projection_model = tf.keras.Model(inputs=proj_input, outputs=proj_function)
        
        return projection_model
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_features': self.n_features,
            'f_r_units': self.f_r_units,
            'f_o_units': self.f_o_units,
            'phi_C_units': self.phi_C_units,
            'proj_units': self.proj_units,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def create_simclr_model(
        n_features,
        f_r_units, 
        f_o_units,
        phi_C_units,
        proj_units,
        temp,
        learning_rate,
):
    """
    Factory method to create and compile a SimCLR model

    Args:
        n_features (int): Number of features (1st dimension of sample)
        f_r_units (tuple of int): Size of each layer in the F_R function in the GNN
        f_o_units (tuple of int): Size of each layer in the F_O function in the GNN
        phi_C_units (tuple of int):
            Size of each layer in the Phi_C function in the GNN. The last entry in this tuple will become
            the dimension of the embeddings produced by the encoder.
        proj_units (tuple of int):
            Size of each layer in the Projection Head
    """
    model = SimCLRModel(
        n_features,
        f_r_units, 
        f_o_units,
        phi_C_units,
        proj_units,
    )

    loss_fn = SimCLRNTXentLoss(temp)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn
    )
    
    return model
