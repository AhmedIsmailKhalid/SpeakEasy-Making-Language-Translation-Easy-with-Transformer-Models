import torch

def PositionalEncoding(model_dim, seq_length):
    pos_encoding = torch.zeros(seq_length, model_dim)
    position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
    divisor  = torch.exp(torch.arange(0, model_dim, 2).float() * -1 * torch.log(torch.tensor(10000.0)) / model_dim)
    pos_encoding[:, 0::2] = torch.sin(position * divisor )
    pos_encoding[:, 1::2] = torch.cos(position * divisor )
    return pos_encoding
