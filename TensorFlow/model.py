import tensorflow as tf
from encoder import Encoder
from decoder import Decoder
from tensorflow import math, cast, float32, linalg, ones, maximum, newaxis
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class TransformerModel(Model):
    def __init__(self, enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length,
                       num_heads, keys_dim, values_dim, model_dim, feedforward_inner_dim, num_layers, dropout_rate, **kwargs):
        super().__init__(**kwargs)

        # Set up the encoder
        self.encoder = Encoder(enc_vocab_size, enc_seq_length, num_heads, keys_dim, values_dim,
                               model_dim, feedforward_inner_dim, num_layers, dropout_rate)

        # Set up the decoder
        self.decoder = Decoder(dec_vocab_size, dec_seq_length, num_heads, keys_dim, values_dim,
                               model_dim, feedforward_inner_dim, num_layers, dropout_rate)

        # Define the final dense layer
        self.model_output_layer = Dense(dec_vocab_size)

    def mask_padding(self, input):
        # Create a mask that marks the zero-padding values in the input as 1.0
        mask = tf.math.equal(input, 0)
        mask = tf.cast(mask, tf.float32)

        # The shape of the mask should be broadcastable to the shape
        # of the attention weights that it will be masking later on
        return mask[:, newaxis, newaxis, :]

    def lookahead_mask(self, shape):
        # Mask out future entries by marking them with a 1.0
        mask = 1 - tf.linalg.band_part(tf.ones((shape, shape)), -1, 0)

        return mask


    def call(self, encoder_input, decoder_input, training):

        # Create padding mask to mask the encoder inputs and the encoder
        # outputs in the decoder
        enc_padding_mask = self.mask_padding(encoder_input)

        # Create and combine padding and look-ahead masks to be fed into the decoder
        dec_in_padding_mask = self.mask_padding(decoder_input)
        dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
        dec_in_lookahead_mask = tf.maximum(dec_in_padding_mask, dec_in_lookahead_mask)

        # Feed the input into the encoder
        encoder_output = self.encoder(encoder_input, enc_padding_mask, training)

        # Feed the encoder output into the decoder
        decoder_output = self.decoder(decoder_input, encoder_output,
                                      dec_in_lookahead_mask, enc_padding_mask, training)

        # Pass the decoder output through a final dense layer
        model_output = self.model_output_layer(decoder_output)

        return model_output
