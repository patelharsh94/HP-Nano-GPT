import torch

# --- Hyperparameters ---
batch_size = 64      # Number of sequences processed in parallel
block_size = 256     # Maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384         # Embedding dimension
n_head = 6           # Number of attention heads
n_layer = 6          # Number of transformer blocks
dropout = 0.2        # Dropout rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# --------------------