This is the implementation of **Self-Supervised Learning (SSL)** for Blackbox using the **SimCSE (Simple Contrastive Learning of Sentence Embeddings)** technique.

This replaces the "dumb" Autoencoder with a "smart" Transformer that understands that `User Admin` and `User Root` are semantically similar, drastically reducing false positives.

### **1. The Loss Function (`src/loss/nt_xent.py`)**
**Role:** The Teacher. It forces the model to pull similar logs together and push different logs apart.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i: Embeddings from first pass (Batch, Dim)
        z_j: Embeddings from second pass (Batch, Dim) with different Dropout
        """
        batch_size = z_i.shape[0]

        # 1. Normalize vectors (Cosine Similarity requires normalized inputs)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # 2. Concatenate representations
        # shape: (2*Batch, Dim)
        z_cat = torch.cat([z_i, z_j], dim=0)

        # 3. Compute Similarity Matrix
        # shape: (2*Batch, 2*Batch)
        sim_matrix = torch.matmul(z_cat, z_cat.T) / self.temperature

        # 4. Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z_i.device)
        sim_matrix.masked_fill_(mask, -9e15)

        # 5. Select Positives
        # For item k, the positive is at k + batch_size (or k - batch_size)
        positives_i = torch.diag(sim_matrix, batch_size)
        positives_j = torch.diag(sim_matrix, -batch_size)
        positives = torch.cat([positives_i, positives_j], dim=0)

        # 6. Calculate Cross Entropy
        # The goal: Maximize the 'positives' value against the sum of all others
        denominator = torch.logsumexp(sim_matrix, dim=1)
        loss = torch.mean(denominator - positives)

        return loss
```

---

### **2. The Model (`src/models/transformer_ssl.py`)**
**Role:** The Brain. A Transformer Encoder that learns context.

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LogTransformerSSL(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super(LogTransformerSSL, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=1) # 1=[PAD]
        self.pos_encoder = PositionalEncoding(d_model)

        # The Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout, # Critical for SimCSE!
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Projection Head (Mapping to contrastive space)
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x, mask=None):
        # x: [Batch, Seq_Len] (Integers)

        # 1. Embed & Positional Encode
        x = self.embedding(x) * math.sqrt(x.size(-1))
        x = self.pos_encoder(x)

        # 2. Transformer Pass
        # We pass padding mask if provided to ignore [PAD] tokens
        out = self.transformer(x, src_key_padding_mask=mask)

        # 3. Pooling (Take the CLS token or Mean Pooling)
        # Here we take Mean Pooling over the sequence
        # out: [Batch, Seq_Len, Dim] -> [Batch, Dim]
        pooled = torch.mean(out, dim=1)

        # 4. Project
        projected = self.projection(pooled)

        # Return projected (for Training loss) and pooled (for Inference)
        return projected, pooled
```

---

### **3. The Training Script (`scripts/train_ssl.py`)**
**Role:** The Trainer. Uses the "Twin Pass" technique.

```python
import sys
import os
import torch
import torch.optim as optim
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.transformer_ssl import LogTransformerSSL
from src.loss.nt_xent import NTXentLoss
from src.preprocessing.tokenizer import LogTokenizer

# --- CONFIG ---
DATA_PATH = "data/raw/training_logs.txt"
ARTIFACTS_DIR = "data/artifacts"
VOCAB_SIZE = 5000
SEQ_LEN = 64
BATCH_SIZE = 64
EPOCHS = 10
LR = 3e-4

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {device}")

    # 1. Load & Tokenize Data
    print(">>> Loading Data...")
    with open(DATA_PATH, 'r') as f:
        logs = f.readlines()

    tokenizer = LogTokenizer(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)
    tokenizer.build_vocab(logs)
    tokenizer.save_vocab(f"{ARTIFACTS_DIR}/vocab.txt")

    # Create Tensor
    vectors = [tokenizer.encode(log) for log in logs]
    data_tensor = torch.tensor(vectors, dtype=torch.long).to(device)

    # 2. Initialize Model
    model = LogTransformerSSL(vocab_size=tokenizer.vocab_size).to(device)
    criterion = NTXentLoss(temperature=0.1) # Standard temp for SimCSE
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # 3. Contrastive Training Loop
    print(">>> Starting Self-Supervised Training...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        permutation = torch.randperm(data_tensor.size(0))

        for i in range(0, data_tensor.size(0), BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch = data_tensor[indices]

            # Create Padding Mask (1 is PAD)
            padding_mask = (batch == 1)

            optimizer.zero_grad()

            # --- THE MAGIC HAPPENS HERE ---
            # Pass the SAME batch twice.
            # The Dropout inside the Transformer creates two slightly different views (z_i, z_j).
            z_i, _ = model(batch, padding_mask)
            z_j, _ = model(batch, padding_mask)

            # Calculate Contrastive Loss
            loss = criterion(z_i, z_j)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"    Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss / (len(data_tensor)/BATCH_SIZE):.4f}")

    # 4. Export to ONNX (For C++ Core)
    print(">>> Exporting to ONNX...")
    model.eval()
    dummy_input = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN)).to(device)
    dummy_mask = (dummy_input == 1)

    # We export a version that only returns 'pooled' (the embedding), not 'projected'
    # We need to wrap it slightly or rely on dynamic axes

    torch.onnx.export(
        model,
        (dummy_input, dummy_mask),
        f"{ARTIFACTS_DIR}/transformer_ssl.onnx",
        input_names=['input_ids', 'padding_mask'],
        output_names=['projected', 'embedding'], # We want 'embedding' for anomaly detection
        dynamic_axes={'input_ids': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}}
    )
    print(">>> DONE.")

if __name__ == "__main__":
    main()
```

---

### **How to use this in `blackbox-core` (C++)**

You are no longer looking for Reconstruction Error (Input vs Output). You are now looking for **Distance from Normal**.

1.  **Offline (Python):** After training, calculate the "Mean Embedding" (Center) of all your normal logs. Save this vector (size 128) as `center_vec.txt`.
2.  **Online (C++ InferenceEngine):**
    *   Load `transformer_ssl.plan` (converted from ONNX).
    *   Pass the log vector. Get the Output Embedding (128 floats).
    *   Calculate **Cosine Distance** between Output and Center.
    *   `Score = 1.0 - CosineSimilarity(Output, Center)`.
    *   If `Score > Threshold`, it is semantically far from normal -> **Anomaly**.

This makes your AI incredibly robust against attackers trying to "fuzz" your logs.