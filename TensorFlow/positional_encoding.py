import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Layer

# Define a class for position embedding with fixed weights
class FixedWeightedPositionEncoding(Layer):
    # Initialize the class with sequence length, vocabulary size, and output dimension
    def __init__(self, seq_length, vocab_size, output_dim, **kwargs):
        super().__init__(**kwargs)
        # Using a dictionary to initialize and hold the embedding layers
        self.embedding_layers = {
            'word': Embedding(
                input_dim=vocab_size, output_dim=output_dim,
                weights=[self.position_encoding(vocab_size, output_dim)],
                trainable=False
            ),
            'position': Embedding(
                input_dim=seq_length, output_dim=output_dim,
                weights=[self.position_encoding(seq_length, output_dim)],
                trainable=False
            )
        }


    # Function to create a position encoding matrix with shape (sequence length, output dimension)
    def position_encoding(self, seq_length, dimension, n=10000):
        # Initialize an empty position encoding matrix
        pos_encoding = np.zeros((seq_length, dimension))

        # Using list comprehension to generate values for the position encoding matrix
        pos_encoding = np.array([
            [np.sin(pos / np.power(n, 2 * i / dimension)) if i % 2 == 0 else np.cos(pos / np.power(n, 2 * i / dimension))
             for i in range(dimension)] for pos in range(seq_length)
        ])

        # Return the position encoding matrix
        return pos_encoding


    # Method to call the layer with an input
    def __call__(self, inputs):
        position_indices = tf.range(tf.shape(inputs)[-1])
        # Using dictionary values to get the embeddings and add them
        embedded = sum([layer(inputs if name == 'word' else position_indices) 
                        for name, layer in self.embedding_layers.items()])
        return embedded
