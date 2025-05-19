"""
Module Name: loss_functions

Description:
    Custom Tensorflow loss functions for unsupervised contrastive learning

Usage:
Author:
Date:
License:
"""
import tensorflow as tf


class SimCLRNTXentLoss(tf.keras.losses.Loss):
    """
    SimCLR loss implementation
    """
    def __init__(self, temp):
        """
        Constructor

        Args:
            temp: Temperature parameter
        """
        super().__init__()
        self.temp = temp

    def call(self, y_true, y_pred):
        # Unpack embeddings
        anchor = y_pred[:, 0]
        positive = y_pred[:, 1]
        batch_size = tf.shape(anchor)[0]

        # Normalize embeddings
        anchor = tf.math.l2_normalize(anchor, axis=1)
        positive = tf.math.l2_normalize(positive, axis=1)

        # Calculate denominator for each sample
        representations = tf.concat([anchor, positive], axis=0)  # [2N, D]
        sim_matrix = tf.matmul(representations, representations, transpose_b=True)  # [2N, 2N]
        self_compare_mask = ~tf.eye(2 * batch_size, dtype=tf.bool)
        sim_exp = tf.exp(sim_matrix / self.temp)
        sim_exp = tf.where(self_compare_mask, sim_exp, 0.)  # Zero out self-comparisons
        denominator = tf.reduce_sum(sim_exp, axis=1)  # [2N]

        # Calculate numerator for each sample
        pos_sim = tf.reduce_sum(anchor * positive, axis=1)  # [N, D] -> [N]
        pos_exp = tf.exp(pos_sim / self.temp)
        numerator = tf.concat([pos_exp, pos_exp], axis=0)  # [2N]

        # Final NT-Xent loss
        loss = -tf.reduce_mean(tf.math.log(numerator / (denominator + 1e-8)))
        return loss

    def get_config(self):
        return {"temp": self.temp}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# from tensorflow.keras.models import load_model

# custom_objects = {"SimCLRNTXentLoss": SimCLRNTXentLoss}

# loaded_model = load_model("simclr_model", custom_objects=custom_objects)


def vicreg_loss(y_true, y_pred, inv_weight=25.0, var_weight=25.0, cov_weight=1.0, gamma=1.0):
    """VICReg loss implementation
    Args:
        y_true: Unused label tensor
        y_pred: Tuple of (embedding1, embedding2) from model
        inv_weight: Weight of invariance (similarity) term in loss function
        var_weight: Weight of variance term in loss function
        cov_weight: Weight of covariance term in loss function
        gamma: Constant target value used for hinge loss in variance term
    """
    embedding1, embedding2 = y_pred[0], y_pred[1]
    batch_size = tf.shape(embedding1)[0]
    embed_dim = tf.shape(embedding1)[1]
    
    # Invariance (similarity) loss
    inv_loss = tf.reduce_mean(tf.reduce_sum(tf.square(embedding1 - embedding2), axis=1))
    
    # Variance loss
    j_std_embed1 = tf.sqrt(tf.math.reduce_variance(embedding1, axis=0) + 1e-8)
    var_loss_1 = tf.reduce_mean(tf.maximum(0.0, gamma - j_std_embed1))
    
    j_std_embed2 = tf.sqrt(tf.math.reduce_variance(embedding2, axis=0) + 1e-8)
    var_loss_2 = tf.reduce_mean(tf.maximum(0.0, gamma - j_std_embed2))

    # Covariance loss
    embed_1_centered = embedding1 - tf.reduce_mean(embedding1, axis=0)
    cov_1 = tf.matmul(embed_1_centered, embed_1_centered, transpose_a=True) / (tf.cast(batch_size, tf.float32) - 1)
    cov_1 = tf.linalg.set_diag(cov_1, tf.zeros(tf.shape(cov_1)[0]))
    cov_loss_1 = tf.reduce_sum(tf.square(cov_1)) / tf.cast(embed_dim, tf.float32)

    embed_2_centered = embedding2 - tf.reduce_mean(embedding2, axis=0)
    cov_2 = tf.matmul(embed_2_centered, embed_2_centered, transpose_a=True) / (tf.cast(batch_size, tf.float32) - 1)
    cov_2 = tf.linalg.set_diag(cov_2, tf.zeros(tf.shape(cov_2)[0]))
    cov_loss_2 = tf.reduce_sum(tf.square(cov_2)) / tf.cast(embed_dim, tf.float32)
    
    # Combined loss
    total_loss = (
        inv_weight * inv_loss +
        var_weight * (var_loss_1 + var_loss_2) +
        cov_weight * (cov_loss_1 + cov_loss_2)
    )
    return total_loss