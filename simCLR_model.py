"""SimCLR model implementation for contrastive learning.

This module implements the SimCLR (Simple Framework for Contrastive Learning of
Visual Representations) architecture adapted for particle physics. The model consists
of an encoder (GNN + dense layers) and a projection head. During pretraining, the
projection head outputs are used for contrastive loss. For downstream tasks, only
the encoder embeddings are used.
"""

import tensorflow as tf

from graph_embeddings import *
from loss_functions import SimCLRNTXentLoss


class SimCLRModel(tf.keras.Model):
    """SimCLR model for self-supervised contrastive learning.

    Attributes:
        n_features: Int, number of particle features.
        f_r_units: List of int, hidden layer sizes for GNN edge function.
        f_o_units: List of int, hidden layer sizes for GNN node function.
        phi_C_units: List of int, hidden layer sizes for encoder dense layers.
        proj_units: List of int, hidden layer sizes for projection head.
        encoder: tf.keras.Model, GNN encoder producing embeddings.
        projection_head: tf.keras.Model, projection head for contrastive loss.
    """

    def __init__(
        self, n_features, f_r_units, f_o_units, phi_C_units, proj_units, **kwargs
    ):
        """Initialize SimCLR model.

        Args:
            n_features: Int, number of features per particle.
            f_r_units: List of int, GNN edge function layer sizes.
            f_o_units: List of int, GNN node function layer sizes.
            f_o_units: List of int, GNN node function layer sizes.
            phi_C_units: List of int, encoder dense layer sizes. Last value is
                embedding dimension.
            proj_units: List of int, projection head layer sizes.
            **kwargs: Additional keyword arguments for tf.keras.Model.
        """
        super().__init__(**kwargs)

        # Store configurations
        self.n_features = n_features
        self.f_r_units = f_r_units
        self.f_o_units = f_o_units
        self.phi_C_units = phi_C_units
        self.proj_units = proj_units

        # Create models
        self.encoder = self._create_encoder_model(
            self.n_features, self.f_r_units, self.f_o_units, self.phi_C_units
        )

        self.projection_head = self._create_projection_head(self.proj_units)

    def build(self, input_shape):
        """Build model by initializing encoder and projection head.

        Args:
            input_shape: Tuple of two shapes for (view1, view2) inputs.
        """
        # Two inputs [view1, view2], each with shape (batch_size, n_features, None)
        self.encoder.build(input_shape[0])
        self.projection_head.build((None, self.phi_C_units[-1]))

        super().build(input_shape)

    def call(self, inputs):
        """Forward pass through encoder and projection head for both views.

        Args:
            inputs: Tuple (view1, view2) of augmented input tensors, each of
                shape (batch_size, n_features, n_particles).

        Returns:
            Tensor of shape (batch_size, 2, proj_dim) containing projection
            head outputs for both views, stacked along dimension 1.
        """
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

    def _create_encoder_model(self, n_features, f_r_units, f_o_units, phi_C_units):
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
        O_Bar = GraphEmbeddings(f_r_units=f_r_units, f_o_units=f_o_units)(input)

        # Trainable function phi_C to transform GNN output to final embeddings
        phi_C = tf.keras.Sequential(
            [
                layer
                for units in phi_C_units
                for layer in [
                    tf.keras.layers.Dense(units),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                ]
            ]
        )(O_Bar)

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
        proj_input = tf.keras.layers.Input(
            shape=(self.phi_C_units[-1],), dtype=tf.float32
        )

        proj_function = tf.keras.Sequential(
            [
                layer
                for units in proj_units
                for layer in [
                    tf.keras.layers.Dense(units),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                ]
            ]
        )(proj_input)

        projection_model = tf.keras.Model(inputs=proj_input, outputs=proj_function)

        return projection_model

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_features": self.n_features,
                "f_r_units": self.f_r_units,
                "f_o_units": self.f_o_units,
                "phi_C_units": self.phi_C_units,
                "proj_units": self.proj_units,
            }
        )
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
    """Create and compile a SimCLR model with NT-Xent loss.

    Factory function to instantiate a SimCLRModel with specified architecture
    and compile it with Adam optimizer and SimCLR contrastive loss.

    Args:
        n_features: Int, number of features per particle.
        f_r_units: List of int, GNN edge function layer sizes.
        f_o_units: List of int, GNN node function layer sizes.
        phi_C_units: List of int, encoder dense layer sizes. The last value
            determines the embedding dimension.
        proj_units: List of int, projection head layer sizes.
        temp: Float, temperature parameter for NT-Xent loss (typically 0.1).
        learning_rate: Float, learning rate for Adam optimizer.

    Returns:
        Compiled SimCLRModel ready for training.

    Example:
        >>> model = create_simclr_model(6, [96, 64], [128, 96], [32, 16],
        ...                              [24, 32], temp=0.1, learning_rate=0.0001)
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss_fn
    )

    return model
