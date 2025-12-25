"""
Module Name: transformation

Description:
    This module provides generators that augment the inputs in a dataset, for use in Contrastive Learning
    and evaluation of model performance on distorted inputs. ViewPairsGenerator creates two
    independently-augmented views of the same input. ViewTransformedGenerator outputs a single augmentation
    of the input. The preferred way to use these generators is through the associated helper functions, which
    create a pipeline through which the original Tensorflow Dataset is augmented.

Usage:
Author:
Date:
License:
"""

from enum import Enum

import numpy as np
import tensorflow as tf


class TransformationType(Enum):
    """Enum for available data transformations"""

    IDENTITY = "identity"
    DELETE_PARTICLE = "delete_particle"


def _delete_particle(x, **kwargs):
    """
    Randomly delete particles from the last dimension using vectorized operations

    Args:
        x (tf.Tensor): Shape of (None, num_features, num_particles)
        **kwargs: Keyword arguments. Supports:
            - num_particles_to_delete (int): Number of particles to delete (default: 1)

    Returns:
        tf.Tensor: Shape of (None, num_features, num_particles - num_particles_to_delete)
    """
    num_particles_to_delete = kwargs.get("num_particles_to_delete", 1)

    x_np = x.numpy()
    batch_size, num_features, num_particles = x_np.shape

    # Ensure we don't try to delete more particles than available
    num_to_delete = min(num_particles_to_delete, num_particles - 1)

    result = np.zeros((batch_size, num_features, num_particles - num_to_delete))

    for i in range(batch_size):
        # Randomly select multiple unique indices to delete
        indices_to_delete = np.random.choice(
            num_particles, size=num_to_delete, replace=False
        )

        # Create a mask for the randomly selected particles
        mask = np.ones(num_particles, dtype=bool)
        mask[indices_to_delete] = False
        # Apply the mask to keep all particles except those in indices_to_delete
        result[i] = x_np[i][:, mask]

    return tf.convert_to_tensor(result)


def _identity(x, **kwargs):
    """
    No transformation

    Args:
        x: Input tensor
        **kwargs: Ignored keyword arguments (for consistency with other transformations)
    """
    return x


# Map transformation enum values to their corresponding functions
TRANSFORMATION_MAP = {
    TransformationType.IDENTITY: _identity,
    TransformationType.DELETE_PARTICLE: _delete_particle,
}


class ViewPairsGenerator:
    def __init__(self, dataset, transformations=None, transformation_kwargs=None):
        """
        Constructor

        Args:
            dataset (tf.data.dataset): A Tensorflow Dataset that's batched
            transformations (list of TransformationType): List of transformations to apply.
                If None, uses all available transformations.
            transformation_kwargs (dict): Optional keyword arguments to pass to all transformation functions.
                Each transformation will extract the parameters it needs.
                Example: {'num_particles_to_delete': 3}
        """
        self.dataset = dataset
        self.transformation_kwargs = transformation_kwargs or {}

        # Build transformation list based on user input
        if transformations is None:
            transformations = list(TransformationType)

        self.TRANSFORMATIONS = [TRANSFORMATION_MAP[t] for t in transformations]

    def generate(self):
        """
        Generate batches with transformed pairs. Use with TF's from_generator API.
        """
        for x_batch, y_batch in self.dataset:
            # Create two views with random transformations
            transform1 = np.random.choice(self.TRANSFORMATIONS)
            view1 = transform1(x_batch, **self.transformation_kwargs)

            transform2 = np.random.choice(self.TRANSFORMATIONS)
            view2 = transform2(x_batch, **self.transformation_kwargs)

            # Yield transformed views and original labels
            yield ((view1, view2), y_batch)


class ViewTransformedGenerator:
    """
    Data generator that applies random transformations to create transformed views
    """

    def __init__(self, dataset, transformations=None, transformation_kwargs=None):
        """
        Constructor

        Args:
            dataset (tf.data.dataset):
                A Tensorflow Dataset that's batched and has signature (None, n_features, None) for X and
                (None, 1) for y.
            transformations (list of TransformationType): List of transformations to apply.
                If None, uses all available transformations.
            transformation_kwargs (dict): Optional keyword arguments to pass to all transformation functions.
                Each transformation will extract the parameters it needs.
                Example: {'num_particles_to_delete': 3}
        """
        self.dataset = dataset
        self.transformation_kwargs = transformation_kwargs or {}

        # Build transformation list based on user input
        if transformations is None:
            transformations = list(TransformationType)

        self.TRANSFORMATIONS = [TRANSFORMATION_MAP[t] for t in transformations]

    def generate(self):
        """Generate batches with randomly transformed data"""
        for x_batch, y_batch in self.dataset:
            # Create a view with a random transformation
            transform = np.random.choice(self.TRANSFORMATIONS)
            transformed_view = transform(x_batch, **self.transformation_kwargs)

            # Yield transformed view and original labels
            yield (transformed_view, y_batch)


def create_transformed_pairs_dataset(
    X, y, batchsize, n_features, transformations=None, transformation_kwargs=None
):
    """
    Create a TensorFlow dataset with transformed pairs for contrastive learning

    Args:
        X: Input features
        y: Target labels
        batchsize: Batch size for the dataset
        n_features: Number of features
        transformations (list of TransformationType): List of transformations to apply.
            If None, uses all available transformations.
        transformation_kwargs (dict): Optional keyword arguments to pass to all transformation functions.
            Each transformation will extract the parameters it needs.
            Example: {'num_particles_to_delete': 3}

    Returns:
        tf.data.Dataset: A repeating dataset of transformed pairs
    """
    # Create TensorFlow datasets
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batchsize)

    view_pairs_generator = ViewPairsGenerator(
        dataset,
        transformations=transformations,
        transformation_kwargs=transformation_kwargs,
    )

    output_signature = (
        (
            tf.TensorSpec(shape=(None, n_features, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, n_features, None), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
    )

    transformed_dataset = tf.data.Dataset.from_generator(
        view_pairs_generator.generate, output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE)

    return transformed_dataset.repeat()


def create_transformed_dataset(
    X, y, batchsize, n_features, transformations=None, transformation_kwargs=None
):
    """
    Create a TensorFlow dataset with random transformations

    Args:
        X: Input features
        y: Target labels
        batchsize: Batch size for the dataset
        n_features: Number of features
        transformations (list of TransformationType): List of transformations to apply.
            If None, uses all available transformations.
        transformation_kwargs (dict): Optional keyword arguments to pass to all transformation functions.
            Each transformation will extract the parameters it needs.
            Example: {'num_particles_to_delete': 3}

    Returns:
        tf.data.Dataset: A repeating dataset of transformed data
    """
    # Create TensorFlow datasets
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batchsize)

    view_transformed_generator = ViewTransformedGenerator(
        dataset,
        transformations=transformations,
        transformation_kwargs=transformation_kwargs,
    )

    output_signature = (
        tf.TensorSpec(shape=(None, n_features, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
    )

    transformed_dataset = tf.data.Dataset.from_generator(
        view_transformed_generator.generate, output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE)

    return transformed_dataset.repeat()
