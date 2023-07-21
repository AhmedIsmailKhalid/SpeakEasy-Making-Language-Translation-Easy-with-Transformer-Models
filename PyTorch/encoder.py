import numpy as np
import torch
import torch.nn as nn
from multihead_attention import MultiHeadAttention
from positional_encoding import PositionalEncoding

class FeedForward(nn.Module):
    def __init__(self, model_dim, feedforward_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(model_dim, feedforward_dim)
        self.fc2 = nn.Linear(feedforward_dim, model_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.nn.functional.relu(out)
        out = self.fc2(out)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, feedforward_dim, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(model_dim, num_heads)
        self.ffn = FeedForward(model_dim, feedforward_dim)

        self.layernorm1 = nn.LayerNorm(model_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(model_dim, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, model_dim)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, model_dim)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, model_dim)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, model_dim)

        return out2

class Encoder(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, feedforward_dim, input_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.model_dim = model_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_vocab_size, model_dim)
        self.pos_encoding = PositionalEncoding(model_dim, maximum_position_encoding)

        self.enc_layers = nn.ModuleList([EncoderLayer(model_dim, num_heads, feedforward_dim, dropout_rate) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout_rate)

    
    def forward(self, x, mask):
        seq_len = x.shape[1]
        
        # Adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, model_dim)
        x *= torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float32))
        x += self.pos_encoding[:seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        return x  # (batch_size, input_seq_len, model_dim)
