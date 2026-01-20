# embedding.py
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Model: uses sentence-transformers MiniLM; we will do attention-aware mean pooling
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def mean_pooling(outputs, attention_mask):
    """
    Attention-aware mean pooling: sum token embeddings weighted by attention mask, then divide by valid tokens.
    """
    token_embeddings = outputs.last_hidden_state  # (batch_size, seq_len, hidden)
    mask = attention_mask.unsqueeze(-1)          # (batch_size, seq_len, 1)
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return (summed / counts)

def get_embedding(text_chunk):
    """
    Return a normalized float32 numpy vector for the input text_chunk.
    Normalized so that ||embedding|| == 1 (for inner-product = cosine).
    """
    # Tokenize single example; keep it simple (not batched here)
    inputs = tokenizer(text_chunk, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    emb = mean_pooling(outputs, inputs['attention_mask']).squeeze(0).cpu().numpy()
    # Ensure float32
    emb = emb.astype("float32")
    # Normalize (L2) for cosine similarity when using IndexFlatIP
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb

def print_embedding_info(embedding):
    print("\n--- EMBEDDING INFORMATION ---")
    print("Type:", type(embedding))
    print("Vector length:", len(embedding))
    print("Shape:", embedding.shape)
    print("First 10 values:", embedding[:10])
    print("-----------------------------\n")
