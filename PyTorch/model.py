import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, feedforward_dim, input_vocab_size, 
                 target_vocab_size, pe_input, pe_target, dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, model_dim, num_heads, feedforward_dim, 
                               input_vocab_size, pe_input, dropout_rate)

        self.decoder = Decoder(num_layers, model_dim, num_heads, feedforward_dim, 
                               target_vocab_size, pe_target, dropout_rate)

        self.final_layer = nn.Linear(model_dim, target_vocab_size)

    def forward(self, inp, tar, enc_padding_mask=None, 
                look_ahead_mask=None, dec_padding_mask=None):

        enc_output = self.encoder(inp, enc_padding_mask)  # (batch_size, inp_seq_len, model_dim)

        # dec_output.shape == (batch_size, tar_seq_len, model_dim)
        dec_output = self.decoder(
            tar, enc_output, look_ahead_mask, dec_padding_mask)


        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output

