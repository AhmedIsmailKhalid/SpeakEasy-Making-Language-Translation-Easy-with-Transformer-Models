import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Layer, Dense, ReLU, Dropout
from multihead_attention import MultiHeadAttention
from positional_encoding import FixedWeightedPositionEncoding

# Implementing the Add & Norm Layer
class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = LayerNormalization()

    def call(self, x, sublayer_x):
        #norm_input = x + sublayer_x
        #return self.layer_norm(norm_input)
        norm_input = tf.add(x, sublayer_x)
        return self.layer_norm(norm_input)

# Implementing the Feed-Forward Layer
class FeedForward(Layer):
    def __init__(self, feedforward_dim, model_dim, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = Dense(feedforward_dim)
        self.dense2 = Dense(model_dim)
        self.activation = ReLU()

    def call(self, x):
        x = self.dense1(x)
        x = self.activation(x)
        return self.dense2(x)

# Implementing the Encoder Layer
class EncoderLayer(Layer):
    def __init__(self, num_heads, keys_dim, values_dim, model_dim, feedforward_dim, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.multihead_attention = MultiHeadAttention(num_heads, keys_dim, values_dim, model_dim)
        self.dropout = Dropout(dropout_rate)
        self.add_norm = AddNormalization()
        self.feed_forward = FeedForward(feedforward_dim, model_dim)

    def call(self, x, padding_mask, training):
        # Multi-head attention layer
        multiheaded_attn_output = self.multihead_attention(x, x, x, padding_mask)

        # Add in a dropout layer
        multiheaded_attn_output = self.dropout(multiheaded_attn_output, training=training)

        # Followed by an Add & Norm layer
        addnorm_output = self.add_norm(x, multiheaded_attn_output)

        # Followed by a fully connected layers
        feedforward_output = self.feed_forward(addnorm_output)
        feedforward_output = self.dropout(feedforward_output, training=training)
        
        return self.add_norm(addnorm_output, feedforward_output)

# Implementing the Encoder
class Encoder(Layer):
    def __init__(self, vocab_size, sequence_length, num_heads, keys_dim, values_dim, model_dim, feedforward_dim, num_layers, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.position_encoding = FixedWeightedPositionEncoding(sequence_length, vocab_size, model_dim)
        self.dropout = Dropout(dropout_rate)
        self.encoder_layer = [EncoderLayer(num_heads, keys_dim, values_dim, model_dim, feedforward_dim, dropout_rate) for _ in range(num_layers)]

    def call(self, input_sentence, padding_mask, training):
        encoded_sentence = self.position_encoding(input_sentence)
        dropout_output = self.dropout(encoded_sentence, training=training)

        layer_output = dropout_output
        for encoder_layer in self.encoder_layer: 
            layer_output = encoder_layer(layer_output, padding_mask, training)
        
        return layer_output
