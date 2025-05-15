import numpy as np
import tensorflow as tf


class ViewPairsGenerator:
    """Data generator that applies random transformations to create view pairs"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.TRANSFORMATIONS = [
            self.identity,
            self.delete_particle,
        ]
    
    def generate(self):
        """Generate batches with transformed pairs"""
        for x_batch, y_batch in self.dataset:
            # Create two views with random transformations
            transform1 = np.random.choice(self.TRANSFORMATIONS)
            view1 = transform1(x_batch)
            transform2 = np.random.choice(self.TRANSFORMATIONS)
            view2 = transform2(x_batch)
            
            # Yield transformed views and original labels
            yield ((view1, view2), y_batch)


    def delete_particle(self, x):
        """
        Randomly delete a particle from the last dimension using vectorized operations
        
        Parameters:
            x: array with shape (None, num_features, num_particles)
        
        Returns:
            array with shape (None, num_features, num_particles - 1)
        """
        x_np = x.numpy()
        batch_size, num_features, num_particles = x_np.shape
    
        indices_to_delete = np.random.randint(0, num_particles, size=batch_size)
        
        result = np.zeros((batch_size, num_features, num_particles-1))
        
        for i in range(batch_size):
            idx = indices_to_delete[i]
            
            # Create a mask for the randomly selected particle
            mask = np.ones(num_particles, dtype=bool)
            mask[idx] = False
            # Apply the mask to keep all particles except the one at idx
            keep_indices = np.where(mask)[0]
            result[i] = x_np[i][:, keep_indices]

        return tf.convert_to_tensor(result)

    def identity(self, x):
        """
        No transformation
        """
        return x


class ViewTransformedGenerator:
    """Data generator that applies random transformations to create transformed views"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.TRANSFORMATIONS = [
            self.identity,
            self.delete_particle,
        ]
    
    def generate(self):
        """Generate batches with randomly transformed data"""
        for x_batch, y_batch in self.dataset:
            # Create a view with a random transformation
            transform = np.random.choice(self.TRANSFORMATIONS)
            transformed_view = transform(x_batch)
            
            # Yield transformed view and original labels
            yield (transformed_view, y_batch)


    def delete_particle(self, x):
        """
        Randomly delete a particle from the last dimension using vectorized operations
        
        Parameters:
            x: array with shape (None, num_features, num_particles)
        
        Returns:
            array with shape (None, num_features, num_particles - 1)
        """
        x_np = x.numpy()
        batch_size, num_features, num_particles = x_np.shape
    
        indices_to_delete = np.random.randint(0, num_particles, size=batch_size)
        
        result = np.zeros((batch_size, num_features, num_particles-1))
        
        for i in range(batch_size):
            idx = indices_to_delete[i]
            
            # Create a mask for the randomly selected particle
            mask = np.ones(num_particles, dtype=bool)
            mask[idx] = False
            # Apply the mask to keep all particles except the one at idx
            keep_indices = np.where(mask)[0]
            result[i] = x_np[i][:, keep_indices]

        return tf.convert_to_tensor(result)

    def identity(self, x):
        """
        No transformation
        """
        return x


def create_transformed_pairs_dataset(X, y, batchsize, n_features):
    # Create TensorFlow datasets
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batchsize)

    view_pairs_generator = ViewPairsGenerator(dataset)

    output_signature = (
        (
            tf.TensorSpec(shape=(None, n_features, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, n_features, None), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    )

    transformed_dataset = tf.data.Dataset.from_generator(
        view_pairs_generator.generate,
        output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE)

    return transformed_dataset.repeat()


def create_transformed_dataset(X, y, batchsize, n_features):
    # Create TensorFlow datasets
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batchsize)

    view_transformed_generator = ViewTransformedGenerator(dataset)

    output_signature = (
        tf.TensorSpec(shape=(None, n_features, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    )

    transformed_dataset = tf.data.Dataset.from_generator(
        view_transformed_generator.generate,
        output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE)

    return transformed_dataset