import torch
import torch.nn as nn
import hyperparameters as hp
from Block import Block
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, hp.n_embd)
        self.position_embedding_table = nn.Embedding(hp.block_size, hp.n_embd)
        self.blocks = nn.Sequential(*[Block(hp.n_embd, n_head=hp.n_head) for _ in range(hp.n_layer)])
        self.ln_f = nn.LayerNorm(hp.n_embd) # final layer norm
        self.lm_head = nn.Linear(hp.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=hp.device))
        x = tok_emb + pos_emb # Add token and position embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -hp.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx