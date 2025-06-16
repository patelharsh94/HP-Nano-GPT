import torch
import hyperparameters as hp
from BigramLanguageModel import BigramLanguageModel

# Getting the tiny shakespeare training data.
# Read the dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# create the vocabulary mapping
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers    
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encode the text into integers
encode = lambda s: [stoi[c] for c in s]
# Decode the integers back into text
decode = lambda l: ''.join([itos[i] for i in l])

# convert the text into a list of integers
data = torch.tensor(encode(text), dtype=torch.long)

# get the train and test data
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Vocabulary size: {vocab_size}")
print(f"Training data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")
print(f"First 100 characters of training data: {train_data[:100]}")

torch.manual_seed(1337)


def get_batch(split):
    """
    Generate a batch of data.
    split: 'train' or 'val'
    """
    batch_data = train_data if split == 'train' else val_data
    # Randomly sample batch_size sequences of length block_size
    ix = torch.randint(len(batch_data) - hp.block_size, (hp.batch_size,))
    # x is the input data, y is the target data
    x = torch.stack([batch_data[i:i + hp.block_size] for i in ix])
    y = torch.stack([batch_data[i + 1:i + hp.block_size + 1] for i in ix])
    x, y = x.to(hp.device), y.to(hp.device)
    return x, y


xbatch, ybatch = get_batch('train')
print(f"Input batch shape: {xbatch.shape}")
print(f"Target batch shape: {ybatch.shape}")


model = BigramLanguageModel(vocab_size)
m = model.to(hp.device)

print(f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate)


@torch.no_grad()
def estimate_loss():
    # Helper function to estimate loss on train/val sets
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(hp.eval_iters)
        for k in range(hp.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# The training loop
for iter in range(hp.max_iters):
    if iter % hp.eval_interval == 0 or iter == hp.max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# Generate text from the trained model
print("\n--- Generating Text ---")
context = torch.zeros((1, 1), dtype=torch.long, device=hp.device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_chars)
print("--------------------")
