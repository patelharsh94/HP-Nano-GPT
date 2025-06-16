import torch
import torch.nn as nn
from torch.nn import functional as F
import hyperparameters as hp

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        # These are the three linear layers that create the Q, K, and V vectors.
        self.key = nn.Linear(hp.n_embd, head_size, bias=False)
        self.query = nn.Linear(hp.n_embd, head_size, bias=False)
        self.value = nn.Linear(hp.n_embd, head_size, bias=False)

        # This creates the triangular mask to hide future tokens.
        self.register_buffer('tril', torch.tril(torch.ones(hp.block_size, hp.block_size)))

        # A dropout layer for regularization.
        self.dropout = nn.Dropout(hp.dropout)

    def forward(self, x):
        B, T, C = x.shape # Batch, Time, Channels
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5
        # Apply the mask to hide future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # Convert scores to probabilities
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # Perform weighted aggregation of values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, head_size)
        return out

