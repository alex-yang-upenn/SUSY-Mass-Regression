import tensorflow as tf
import numpy as np

import os
import random

class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batchsize, shuffle=True):
        super().__init__()  # Call parent class's __init__
        
        # Initialize parameters
        self.data_dir = data_dir
        self.batchsize = batchsize
        self.shuffle = shuffle
        
        # Initalize numpy files list
        self.input_chunks = sorted([f for f in os.listdir(data_dir) if "_input_" in f])
        self.output_chunks = sorted([f for f in os.listdir(data_dir) if "_output_" in f])
        
        # Check for data length validity
        self.num_chunks = len(self.input_chunks)
        if self.num_chunks != len(self.output_chunks):
            raise AssertionError("Mismatch in number of input and output chunks")
        
        # Initialize chunk order
        self.chunk_order = list(range(self.num_chunks))
        if self.shuffle:
            random.shuffle(self.chunk_order)

        # Calculate total number of samples and batches
        self.total_samples = self._count_samples()
        
        # Initialize the current chunk data
        self.current_inputs = None
        self.current_outputs = None
        self.current_chunk_idx = 0
        self.current_position = 0
        
        # Load first chunk
        self._load_chunk()
        
    def _count_samples(self):
        """Count total samples across all chunks"""
        total_samples = 0
        for chunk in self.output_chunks:
            chunk_data = np.load(os.path.join(self.data_dir, chunk))
            total_samples += len(chunk_data)
        return total_samples
    
    def _load_chunk(self):
        """Load a new chunk of data"""
        # Accesses the numpy file by its index
        chunk_idx = self.chunk_order[self.current_chunk_idx]
        self.current_inputs = np.load(os.path.join(self.data_dir, self.input_chunks[chunk_idx]))
        self.current_outputs = np.load(os.path.join(self.data_dir, self.output_chunks[chunk_idx]))
        
        # Shuffles within the file
        if self.shuffle:
            indices = np.arange(len(self.current_inputs))
            np.random.shuffle(indices)
            self.current_inputs = self.current_inputs[indices]
            self.current_outputs = self.current_outputs[indices]
        
        # Start at beginning of file
        self.current_position = 0
    
    def __len__(self):
        """Return the number of batches per epoch"""
        return self.total_samples // self.batchsize
    
    def __getitem__(self, idx):
        """Get a batch of data"""
        # If we've exhausted the current chunk, load the next one
        if self.current_position >= len(self.current_inputs):
            self.current_chunk_idx = (self.current_chunk_idx + 1) % self.num_chunks
            self._load_chunk()
        
        # Get the batch
        start_idx = self.current_position
        end_idx = start_idx + self.batchsize
        
        # Handle case where batch spans multiple chunks
        if end_idx > len(self.current_inputs):
            # Get remaining data from current chunk
            inputs_part1 = self.current_inputs[start_idx:]
            outputs_part1 = self.current_outputs[start_idx:]
            
            # Load next chunk and get remaining data
            self.current_chunk_idx = (self.current_chunk_idx + 1) % self.num_chunks
            self._load_chunk()
            
            remaining = self.batchsize - len(inputs_part1)
            inputs_part2 = self.current_inputs[:remaining]
            outputs_part2 = self.current_outputs[:remaining]
            
            # Combine the parts
            batch_inputs = np.concatenate([inputs_part1, inputs_part2])
            batch_outputs = np.concatenate([outputs_part1, outputs_part2])
            
            self.current_position = remaining
        else:
            # Normal case - batch fits in current chunk
            batch_inputs = self.current_inputs[start_idx:end_idx]
            batch_outputs = self.current_outputs[start_idx:end_idx]
            self.current_position = end_idx
        
        return batch_inputs, batch_outputs
    
    def on_epoch_end(self):
        """Called at the end of every epoch"""
        if self.shuffle:
            random.shuffle(self.chunk_order)
        self.current_chunk_idx = 0
        self._load_chunk()