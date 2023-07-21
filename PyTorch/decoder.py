import torch
import torch.nn as nn
from multihead_attention import MultiHeadAttention
from positional_encoding import PositionalEncoding
from encoder import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, feedforward_dim, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(model_dim, num_heads)
        self.mha2 = MultiHeadAttention(model_dim, num_heads)

        self.ffn = FeedForward(model_dim, feedforward_dim)

        self.layernorm1 = nn.LayerNorm(model_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(model_dim, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(model_dim, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, model_dim)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, model_dim)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, model_dim)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, model_dim)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, model_dim)

        return out3, attn_weights_block1, attn_weights_block2

class Decoder(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, feedforward_dim, target_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.model_dim = model_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(target_vocab_size, model_dim)
        self.pos_encoding = PositionalEncoding(model_dim, maximum_position_encoding)

        self.dec_layers = nn.ModuleList([DecoderLayer(model_dim, num_heads, feedforward_dim, dropout_rate) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        seq_len = x.shape[1]
        
        x = self.embedding(x)  # (batch_size, target_seq_len, model_dim)
        x *= torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float32))
        x += self.pos_encoding[:seq_len, :]
        
        x = self.dropout(x)

        for i in range(self.num_layers):
            x, _, _ = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)


        return x  # (batch_size, target_seq_len, model_dim)
