"""Data augmentation transformations for contrastive learning.

This module provides data generators and transformation functions for augmenting
particle physics data. ViewPairsGenerator creates pairs of independently augmented
views for contrastive learning (SimCLR). ViewTransformedGenerator creates single
augmented views for evaluation. Available transformations include particle deletion
and identity (no transformation).
"""

from enum import Enum

import numpy as np
import tensorflow as tf


class TransformationType(Enum):
    """Enum for available data transformations"""

    IDENTITY = "identity"
    DELETE_PARTICLE = "delete_particle"


def _delete_particle(x, **kwargs):
    """Randomly delete particles from input tensor.

    Randomly removes specified number of particles from each sample in the batch.
    Uses vectorized NumPy operations for efficiency. This augmentation simulates
    detector inefficiencies or incomplete particle reconstruction.

    Args:
        x: tf.Tensor of shape (batch_size, n_features, n_particles) containing
            particle features.
        **kwargs: Optional keyword arguments:
            - num_particles_to_delete: Int, number of particles to randomly delete
              per sample. Defaults to 1. Will be clamped to ensure at least one
              particle remains.

    Returns:
        tf.Tensor: Augmented tensor of shape (batch_size, n_features,
            n_particles - num_particles_to_delete) with randomly deleted particles.

    Example:
        >>> x = tf.random.normal((32, 6, 4))  # 32 samples, 6 features, 4 particles
        >>> x_aug = _delete_particle(x, num_particles_to_delete=1)
        >>> print(x_aug.shape)  # (32, 6, 3)
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
    """Identity transformation (no augmentation).

    Returns input unchanged. Used as a transformation option in random
    augmentation pipelines to include non-augmented views.

    Args:
        x: Input tensor of any shape.
        **kwargs: Ignored keyword arguments for consistency with other
            transformation functions.

    Returns:
        Input tensor unchanged.
    """
    return x


# Map transformation enum values to their corresponding functions
TRANSFORMATION_MAP = {
    TransformationType.IDENTITY: _identity,
    TransformationType.DELETE_PARTICLE: _delete_particle,
}


class ViewPairsGenerator:
    """Generator creating pairs of augmented views for contrastive learning.

    Generates two independently augmented views of each input sample by applying
    random transformations. Used for SimCLR contrastive learning where the model
    learns to produce similar embeddings for different views of the same input.
    """

    def __init__(self, dataset, transformations=None, transformation_kwargs=None):
        """Initialize ViewPairsGenerator.

        Args:
            dataset: tf.data.Dataset, a batched TensorFlow dataset with signature
                (X, y) where X has shape (batch_size, n_features, n_particles).
            transformations: List of TransformationType enum values specifying which
                transformations to randomly sample from. If None, uses all available
                transformations.
            transformation_kwargs: Dict of keyword arguments passed to all transformation
                functions. Each transformation extracts the parameters it needs.
                Example: {'num_particles_to_delete': 2}.
        """
        self.dataset = dataset
        self.transformation_kwargs = transformation_kwargs or {}

        # Build transformation list based on user input
        if transformations is None:
            transformations = list(TransformationType)

        self.TRANSFORMATIONS = [TRANSFORMATION_MAP[t] for t in transformations]

    def generate(self):
        """Generate batches with pairs of transformed views.

        Yields:
            tuple: ((view1, view2), y_batch) where view1 and view2 are independently
                augmented versions of the same input batch, and y_batch is the
                original labels.
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
    """Generator creating single augmented views for evaluation.

    Applies random transformations to create single augmented views of each input.
    Used for evaluating model robustness to augmentations or for training models
    that don't require paired views.
    """

    def __init__(self, dataset, transformations=None, transformation_kwargs=None):
        """Initialize ViewTransformedGenerator.

        Args:
            dataset: tf.data.Dataset, a batched TensorFlow dataset with signature
                (X, y) where X has shape (batch_size, n_features, n_particles) and
                y has shape (batch_size, 1).
            transformations: List of TransformationType enum values specifying which
                transformations to randomly sample from. If None, uses all available
                transformations.
            transformation_kwargs: Dict of keyword arguments passed to all transformation
                functions. Each transformation extracts the parameters it needs.
                Example: {'num_particles_to_delete': 2}.
        """
        self.dataset = dataset
        self.transformation_kwargs = transformation_kwargs or {}

        # Build transformation list based on user input
        if transformations is None:
            transformations = list(TransformationType)

        self.TRANSFORMATIONS = [TRANSFORMATION_MAP[t] for t in transformations]

    def generate(self):
        """Generate batches with randomly transformed views.

        Yields:
            tuple: (transformed_view, y_batch) where transformed_view is an augmented
                version of the input batch and y_batch is the original labels.
        """
        for x_batch, y_batch in self.dataset:
            # Create a view with a random transformation
            transform = np.random.choice(self.TRANSFORMATIONS)
            transformed_view = transform(x_batch, **self.transformation_kwargs)

            # Yield transformed view and original labels
            yield (transformed_view, y_batch)


def create_transformed_pairs_dataset(
    X, y, batchsize, n_features, transformations=None, transformation_kwargs=None
):
    """Create TensorFlow dataset with transformed pairs for contrastive learning.

    Creates a repeating tf.data.Dataset that yields pairs of independently augmented
    views of the same input. Designed for SimCLR and other contrastive learning
    approaches.

    Args:
        X: Array of input features, shape (n_samples, n_features, n_particles).
        y: Array of target labels, shape (n_samples, 1).
        batchsize: Int, batch size for the dataset.
        n_features: Int, number of features per particle.
        transformations: List of TransformationType enum values. If None, uses all
            available transformations.
        transformation_kwargs: Dict of parameters for transformations. For example,
            {'num_particles_to_delete': 2}.

    Returns:
        tf.data.Dataset: Repeating dataset yielding ((view1, view2), y) where:
            - view1, view2: Augmented tensors of shape (batch_size, n_features, variable_particles)
            - y: Labels of shape (batch_size, 1)

    Example:
        >>> dataset = create_transformed_pairs_dataset(X_train, y_train, 128, 6,
        ...     transformation_kwargs={'num_particles_to_delete': 1})
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
    """Create TensorFlow dataset with random transformations.

    Creates a repeating tf.data.Dataset that yields single augmented views of
    each input. Used for evaluating model robustness or training with augmentation.

    Args:
        X: Array of input features, shape (n_samples, n_features, n_particles).
        y: Array of target labels, shape (n_samples, 1).
        batchsize: Int, batch size for the dataset.
        n_features: Int, number of features per particle.
        transformations: List of TransformationType enum values. If None, uses all
            available transformations.
        transformation_kwargs: Dict of parameters for transformations. For example,
            {'num_particles_to_delete': 2}.

    Returns:
        tf.data.Dataset: Repeating dataset yielding (transformed_view, y) where:
            - transformed_view: Augmented tensor of shape (batch_size, n_features, variable_particles)
            - y: Labels of shape (batch_size, 1)

    Example:
        >>> dataset = create_transformed_dataset(X_train, y_train, 128, 6,
        ...     transformation_kwargs={'num_particles_to_delete': 1})
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
