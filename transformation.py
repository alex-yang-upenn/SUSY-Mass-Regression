import numpy as np
import tensorflow as tf


class ViewPairsGenerator:
    """Data generator that applies random transformations to create view pairs"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.TRANSFORMATIONS = [
            self.identity,
            self.delete_particle,
            self.reshuffle_particles
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
            yield [view1, view2], y_batch


    def delete_particle(self, x):
        """
        Randomly delete a particle from the last dimension using vectorized operations
        
        Parameters:
            x: array with shape (None, num_features, num_particles)
        
        Returns:
            array with shape (None, num_features, num_particles - 1)
        """
        batch_size, num_features, num_particles = x.shape
        
        # Generate random indices to delete
        indices_to_delete = np.random.randint(0, num_particles, size=batch_size)
        
        # Create a mask array
        mask = np.ones((batch_size, num_particles), dtype=bool)  # (batch_size, num_particles)
        mask[np.arange(batch_size), indices_to_delete] = False
        mask = mask[:, np.newaxis, :]  # (batch_size, 1, num_particles)
        

        result = x[mask].reshape(batch_size, num_features, num_particles-1)        
        return result


    def reshuffle_particles(self, x):
        """
        Randomly change the order of particles along the last dimension
        
        Args:
            x: array with shape (None, num_features, num_particles)
        
        Returns:
            array with the same shape but shuffled particles
        """
        batch_size, num_features, num_particles = x.shape
        
        # Create a shuffled index array for each item in the batch
        idx = np.stack([np.random.permutation(num_particles) for _ in range(batch_size)])  # (batch_size, num_particles)
        
        # Set up indices
        batch_idx = np.arange(batch_size)[:, None, None]  # (batch_size, 1, 1)
        feature_idx = np.arange(num_features)[None, :, None]  # (1, num_features, 1)
        particle_idx = idx[:, None, :]  # (batch_size, 1, num_particles)
        
        # Index: dim_1=batch_idx, dim_2=feature_idx, dim_3=particle_idx
        result = x[batch_idx, feature_idx, particle_idx]
        
        return result

    def identity(self, x):
        """
        No transformation
        """
        return x