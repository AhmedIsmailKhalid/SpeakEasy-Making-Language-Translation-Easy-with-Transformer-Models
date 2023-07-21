from tensorflow.keras.layers import Layer, Dropout
from multihead_attention import MultiHeadAttention
from positional_encoding import FixedWeightedPositionEncoding
from encoder import AddNormalization, FeedForward

# Implementing the Decoder Layer
class DecoderLayer(Layer):
    def __init__(self, num_heads, keys_dim, values_dim, model_dim, feedforward_dim, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.multihead_attention = MultiHeadAttention(num_heads, keys_dim, values_dim, model_dim)
        self.dropout = Dropout(dropout_rate)
        self.add_norm = AddNormalization()
        self.feed_forward = FeedForward(feedforward_dim, model_dim)

    def call(self, x, encoder_output, lookahead_mask, padding_mask, training):
        # Multi-head attention layer
        multihead_attention_output = self.multihead_attention(x, x, x, lookahead_mask)
        multihead_attention_output = self.dropout(multihead_attention_output, training=training)
        add_norm_output = self.add_norm(x, multihead_attention_output)

        # Multi-head attention layer with encoder output
        multihead_attention_output = self.multihead_attention(add_norm_output, encoder_output, encoder_output, padding_mask)
        multihead_attention_output = self.dropout(multihead_attention_output, training=training)
        add_norm_output = self.add_norm(add_norm_output, multihead_attention_output)

        # Feedforward layer
        feed_forward_output = self.feed_forward(add_norm_output)
        feed_forward_output = self.dropout(feed_forward_output, training=training)
        return self.add_norm(add_norm_output, feed_forward_output)

# Implementing the Decoder
class Decoder(Layer):
    def __init__(self, vocab_size, sequence_length, num_heads, keys_dim, values_dim, model_dim, feedforward_dim, num_layers, dropout_rate,
                       **kwargs):
        super().__init__(**kwargs)
        self.position_encoding = FixedWeightedPositionEncoding(sequence_length, vocab_size, model_dim)
        self.dropout = Dropout(dropout_rate)
        self.decoder_layers = [DecoderLayer(num_heads, keys_dim, values_dim, model_dim, feedforward_dim, dropout_rate)
                              for _ in range(num_layers)]

    def call(self, output_target, encoder_output, lookahead_mask, padding_mask, training):
        # Generate the positional encoding
        pos_encoding_output = self.position_encoding(output_target)
        x = self.dropout(pos_encoding_output, training=training)

        # Pass on the positional encoded values to each decoder layer
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, lookahead_mask, padding_mask, training)

        return x
