"""
Module Name: graph_embeddings

Description:
    Custom Tensorflow layer that implements a Graph Neural Network to transform particle data to embeddings.
    Layer architecture:
        - Constructs adjacency matrices based on number of particles in batch
        - Constructs B_pp, concatenation of each edge's sender and receiver features
        - Applies trainable function f_R to each edge in B_pp to obtain E_pp hidden representation
        - Append input and E_pp to create C_pp_D combined representation
        - Applies trainable function f_O to obtain O_post post-interaction representation
        - Sums over O_post columns to obtain embeddings
Usage:
Author:
Date:
License:
"""
import tensorflow as tf


class GraphEmbeddings(tf.keras.layers.Layer):
    def __init__(self,         
                f_r_units, 
                f_o_units,
                **kwargs):
        super().__init__(**kwargs)

        self.output_dim = f_o_units[-1]

        self.f_r_units = f_r_units
        self.f_o_units = f_o_units

        self.f_R = tf.keras.Sequential([
            layer
            for units in self.f_r_units
            for layer in [
                tf.keras.layers.Dense(units),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
            ]
        ])

        self.f_O = tf.keras.Sequential([
            layer
            for units in self.f_o_units
            for layer in [
                tf.keras.layers.Dense(units),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
            ]
        ])

    def build(self, input_shape):
        P = input_shape[1]
        
        self.f_R.build((None, None, 2 * P))
        
        self.f_O.build((None, None, P + self.f_r_units[-1]))
        
        super().build(input_shape)

    @tf.function(reduce_retracing=True)
    def call(self, x, training=None):
        n_particles = tf.shape(x)[2]
        rr, rr_t, rs = self._create_interaction_matrices(n_particles)

        # Build B_pp
        X_dot_RR = tf.matmul(x, rr)     # [batch_size, P, N_O] × [N_O, N_E] -> [batch_size, P, N_E]
        X_dot_RS = tf.matmul(x, rs)     # [batch_size, P, N_O] × [N_O, N_E] -> [batch_size, P, N_E]
        B_pp = tf.concat([X_dot_RR, X_dot_RS], axis=1)  # [batch_size, 2P, N_E]
        
        # Apply f_R to all edges
        B_pp_t = tf.transpose(B_pp, perm=[0, 2, 1])         # [batch_size, N_E, 2P]
        E_pp_t = self.f_R(B_pp_t, training=training)        # [batch_size, N_E, D_E]
        E_pp = tf.transpose(E_pp_t, perm=[0, 2, 1])         # [batch_size, D_E, N_E]
        
        # Create combined representation for all samples
        E_pp_bar = tf.matmul(E_pp, rr_t)                    # [batch_size, D_E, N_E] × [N_E, N_O] -> [batch_size, D_E, N_O]
        C_pp_D = tf.concat([x, E_pp_bar], axis=1)           # [batch_size, P + D_E, N_O]

        # Apply f_O to all nodes in batch
        C_pp_D_t = tf.transpose(C_pp_D, perm=[0, 2, 1])     # [batch_size, N_O, P + D_E]
        O_post_t = self.f_O(C_pp_D_t, training=training)    # [batch_size, N_O, D_O]
        O_post = tf.transpose(O_post_t, perm=[0, 2, 1])     # [batch_size, D_O, N_O]

        # Global pooling over nodes for each sample
        return tf.reduce_sum(O_post, axis=2)                # [batch_size, D_O]
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def _create_interaction_matrices(self, n):       
        i, j = tf.meshgrid(tf.range(n), tf.range(n))    # Adjacency Matrix
        mask = i != j   # All directed edges except for self-loops
        
        receivers = tf.boolean_mask(i, mask)    # Extract all edges targets
        senders = tf.boolean_mask(j, mask)      # Extract all edges sources
        
        n_edges = n * (n - 1)
        
        rr = tf.scatter_nd(     # Convert receivers to one-hot representation
            tf.stack([receivers, tf.range(n_edges)], axis=1),   # Create indices pairs
            tf.ones(n_edges),                                   # Create one 1.0 value per edge
            [n, n_edges]                                        # Shape of output
        )
        
        rs = tf.scatter_nd(     # Convert senders to one-hot representation
            tf.stack([senders, tf.range(n_edges)], axis=1),     # Create indices pairs
            tf.ones(n_edges),                                   # Create one 1.0 value per edge
            [n, n_edges]                                        # Shape of output
        )
        
        return rr, tf.transpose(rr), rs
