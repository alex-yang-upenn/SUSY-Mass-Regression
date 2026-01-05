"""Graph Neural Network layer for particle physics embeddings.

This module implements a custom TensorFlow/Keras layer that processes particle
data as graphs. Each particle is a node, and edges connect all particle pairs.
The layer learns message-passing functions to aggregate information across the
graph and produce fixed-size embeddings via global pooling.
"""

import tensorflow as tf


class GraphEmbeddings(tf.keras.layers.Layer):
    """Graph Neural Network layer using edge-based message passing.

    Implements a GNN that:
    1. Constructs fully-connected graph (all-to-all particle connections)
    2. Applies edge function f_R to learn edge representations
    3. Aggregates edge messages back to nodes
    4. Applies node function f_O for post-interaction representations
    5. Global sum pooling to produce fixed-size embeddings

    Attributes:
        output_dim: Int, dimension of output embeddings.
        f_r_units: List of int, hidden layer sizes for edge function f_R.
        f_o_units: List of int, hidden layer sizes for node function f_O.
        f_R: tf.keras.Sequential, edge message function.
        f_O: tf.keras.Sequential, node update function.
    """

    def __init__(self, f_r_units, f_o_units, **kwargs):
        """Initialize GraphEmbeddings layer.

        Args:
            f_r_units: List of int, hidden layer sizes for edge function.
                For example, [96, 64, 32] creates a 3-layer network.
            f_o_units: List of int, hidden layer sizes for node function.
                The last value determines the output embedding dimension.
            **kwargs: Additional keyword arguments for tf.keras.layers.Layer.
        """
        super().__init__(**kwargs)

        self.output_dim = f_o_units[-1]

        self.f_r_units = f_r_units
        self.f_o_units = f_o_units

        self.f_R = tf.keras.Sequential(
            [
                layer
                for units in self.f_r_units
                for layer in [
                    tf.keras.layers.Dense(units),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                ]
            ]
        )

        self.f_O = tf.keras.Sequential(
            [
                layer
                for units in self.f_o_units
                for layer in [
                    tf.keras.layers.Dense(units),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                ]
            ]
        )

    def build(self, input_shape):
        """Build layer by initializing f_R and f_O with correct input shapes.

        Args:
            input_shape: Tuple, expected input shape (batch_size, n_features, n_particles).
        """
        P = input_shape[1]

        self.f_R.build((None, None, 2 * P))

        self.f_O.build((None, None, P + self.f_r_units[-1]))

        super().build(input_shape)

    @tf.function(reduce_retracing=True)
    def call(self, x, training=None):
        """Forward pass of Graph Neural Network layer.

        Args:
            x: Input tensor of shape (batch_size, n_features, n_particles).
            training: Bool or None, whether in training mode (affects batch norm).

        Returns:
            Tensor of shape (batch_size, output_dim) containing graph embeddings.
        """
        n_particles = tf.shape(x)[2]
        rr, rr_t, rs = self._create_interaction_matrices(n_particles)

        # Build B_pp
        X_dot_RR = tf.matmul(
            x, rr
        )  # [batch_size, P, N_O] × [N_O, N_E] -> [batch_size, P, N_E]
        X_dot_RS = tf.matmul(
            x, rs
        )  # [batch_size, P, N_O] × [N_O, N_E] -> [batch_size, P, N_E]
        B_pp = tf.concat([X_dot_RR, X_dot_RS], axis=1)  # [batch_size, 2P, N_E]

        # Apply f_R to all edges
        B_pp_t = tf.transpose(B_pp, perm=[0, 2, 1])  # [batch_size, N_E, 2P]
        E_pp_t = self.f_R(B_pp_t, training=training)  # [batch_size, N_E, D_E]
        E_pp = tf.transpose(E_pp_t, perm=[0, 2, 1])  # [batch_size, D_E, N_E]

        # Create combined representation for all samples
        E_pp_bar = tf.matmul(
            E_pp, rr_t
        )  # [batch_size, D_E, N_E] × [N_E, N_O] -> [batch_size, D_E, N_O]
        C_pp_D = tf.concat([x, E_pp_bar], axis=1)  # [batch_size, P + D_E, N_O]

        # Apply f_O to all nodes in batch
        C_pp_D_t = tf.transpose(C_pp_D, perm=[0, 2, 1])  # [batch_size, N_O, P + D_E]
        O_post_t = self.f_O(C_pp_D_t, training=training)  # [batch_size, N_O, D_O]
        O_post = tf.transpose(O_post_t, perm=[0, 2, 1])  # [batch_size, D_O, N_O]

        # Global pooling over nodes for each sample
        return tf.reduce_sum(O_post, axis=2)  # [batch_size, D_O]

    def compute_output_shape(self, input_shape):
        """Compute output shape for given input shape.

        Args:
            input_shape: Tuple, input shape (batch_size, n_features, n_particles).

        Returns:
            Tuple: Output shape (batch_size, output_dim).
        """
        return (input_shape[0], self.output_dim)

    def _create_interaction_matrices(self, n):
        """Create interaction matrices for fully-connected particle graph.

        Constructs matrices that encode sender-receiver relationships for all
        edges in a fully-connected graph (excluding self-loops).

        Args:
            n: Int or Tensor, number of particles (nodes) in the graph.

        Returns:
            tuple: Three tensors (rr, rr_t, rs) where:
                - rr: Receiver matrix, shape (n, n*(n-1))
                - rr_t: Transposed receiver matrix, shape (n*(n-1), n)
                - rs: Sender matrix, shape (n, n*(n-1))
        """
        i, j = tf.meshgrid(tf.range(n), tf.range(n))  # Adjacency Matrix
        mask = i != j  # All directed edges except for self-loops

        receivers = tf.boolean_mask(i, mask)  # Extract all edges targets
        senders = tf.boolean_mask(j, mask)  # Extract all edges sources

        n_edges = n * (n - 1)

        rr = tf.scatter_nd(  # Convert receivers to one-hot representation
            tf.stack([receivers, tf.range(n_edges)], axis=1),  # Create indices pairs
            tf.ones(n_edges),  # Create one 1.0 value per edge
            [n, n_edges],  # Shape of output
        )

        rs = tf.scatter_nd(  # Convert senders to one-hot representation
            tf.stack([senders, tf.range(n_edges)], axis=1),  # Create indices pairs
            tf.ones(n_edges),  # Create one 1.0 value per edge
            [n, n_edges],  # Shape of output
        )

        return rr, tf.transpose(rr), rs
