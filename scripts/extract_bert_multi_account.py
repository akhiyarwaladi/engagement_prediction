#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract IndoBERT Features from Multi-Account Dataset
====================================================

Extract 768-dimensional BERT embeddings from 8,610 posts (8 UNJA accounts)

Expected runtime: ~40-60 minutes (CPU), ~10-15 minutes (GPU)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print(" " * 20 + "MULTI-ACCOUNT BERT EXTRACTION")
print(" " * 15 + "IndoBERT Embeddings for 1,579 Posts")
print("=" * 80 + "\n")

# Check PyTorch availability
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    print(f"[OK] PyTorch version: {torch.__version__}")
    print(f"[OK] Device: {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print("[FAIL] Error: PyTorch or Transformers not installed!")
    print("   Run: bash setup_transformers.sh")
    sys.exit(1)

# Load IndoBERT model
print("\n[LOAD] Loading IndoBERT model...")
print("   (This may take a minute on first run)")

model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print(f"[OK] Loaded: {model_name}")
print(f"   Model parameters: ~110M")

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()  # Set to evaluation mode

print(f"[OK] Model ready on {device}")

def extract_bert_embedding(caption: str) -> np.ndarray:
    """
    Extract 768-dimensional BERT embedding from caption.

    Args:
        caption: Instagram caption text (Indonesian)

    Returns:
        768-dim numpy array
    """
    # Handle empty captions
    if not caption or pd.isna(caption):
        return np.zeros(768)

    # Tokenize (max 128 tokens for efficiency)
    inputs = tokenizer(
        str(caption),
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding='max_length'
    )

    # Move to same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get embeddings (no gradient computation)
    with torch.no_grad():
        outputs = model(**inputs)

    # Use [CLS] token embedding (sentence-level representation)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()

    # Convert to numpy
    return cls_embedding.cpu().numpy()

# Load multi-account data
print("\n[DATA] Loading multi-account dataset...")
df = pd.read_csv('multi_account_dataset.csv')
print(f"   Loaded {len(df)} posts")
print(f"   fst_unja: {len(df[df['account'] == 'fst_unja'])} posts")
print(f"   univ.jambi: {len(df[df['account'] == 'univ.jambi'])} posts")

# Extract BERT embeddings
print("\n[BERT] Extracting BERT embeddings from captions...")
print("   This will take ~15-20 minutes on CPU, ~3-5 minutes on GPU")
print("")

bert_embeddings = []
batch_size = 32  # Process in batches for GPU efficiency

import time
start_time = time.time()

for idx in range(0, len(df), batch_size):
    batch_end = min(idx + batch_size, len(df))
    batch_captions = df['caption'].iloc[idx:batch_end].fillna('')

    # Process batch
    for caption in batch_captions:
        embedding = extract_bert_embedding(str(caption))
        bert_embeddings.append(embedding)

    # Progress with ETA
    progress = batch_end / len(df) * 100
    elapsed = time.time() - start_time
    if batch_end > batch_size:
        eta = (elapsed / batch_end) * (len(df) - batch_end)
        print(f"  Progress: {batch_end}/{len(df)} ({progress:.1f}%) | ETA: {eta/60:.1f} min")
    else:
        print(f"  Progress: {batch_end}/{len(df)} ({progress:.1f}%)")

elapsed_total = time.time() - start_time
print(f"\n[TIME] Total extraction time: {elapsed_total/60:.2f} minutes")

# Convert to DataFrame
print("\n[STATS] Creating feature DataFrame...")
bert_features_df = pd.DataFrame(
    bert_embeddings,
    columns=[f'bert_dim_{i}' for i in range(768)]
)

# Add post metadata for reference
bert_features_df.insert(0, 'post_id', df['post_id'].values)
bert_features_df.insert(1, 'account', df['account'].values)
bert_features_df.insert(2, 'caption_preview', df['caption'].fillna('').str[:50])

# Save to CSV
output_path = 'data/processed/bert_embeddings_multi_account.csv'
Path(output_path).parent.mkdir(parents=True, exist_ok=True)

bert_features_df.to_csv(output_path, index=False)
print(f"[SAVE] Saved to: {output_path}")

# Statistics
print("\n[INFO] Embedding Statistics:")
print(f"   Shape: {bert_features_df.shape}")
print(f"   Dimensions: 768")
print(f"   Total posts: {len(bert_features_df)}")
print(f"   fst_unja posts: {len(bert_features_df[bert_features_df['account'] == 'fst_unja'])}")
print(f"   univ.jambi posts: {len(bert_features_df[bert_features_df['account'] == 'univ.jambi'])}")

# Show sample
print("\n[SAMPLE] Sample embeddings (first 5 dimensions):")
sample_cols = ['account', 'caption_preview'] + [f'bert_dim_{i}' for i in range(5)]
print(bert_features_df[sample_cols].head(3))

# Embedding quality check
print("\n[CHECK] Quality Check:")
embeddings_array = bert_embeddings
mean_norm = np.mean([np.linalg.norm(emb) for emb in embeddings_array])
print(f"   Average embedding norm: {mean_norm:.2f}")
print(f"   Expected range: 8-15 (typical for BERT)")

if mean_norm < 5:
    print("   [WARN] Warning: Embedding norms unusually low")
elif mean_norm > 20:
    print("   [WARN] Warning: Embedding norms unusually high")
else:
    print("   [OK] Embeddings look good!")

# Check for variability
std_per_dim = np.std(embeddings_array, axis=0).mean()
print(f"   Average std per dimension: {std_per_dim:.3f}")
print(f"   Expected range: 0.3-1.0")

if std_per_dim < 0.1:
    print("   [WARN] Warning: Low variability, embeddings may be too similar")
else:
    print("   [OK] Good variability across embeddings!")

# Summary
print("\n" + "=" * 80)
print("EXTRACTION COMPLETE! [DONE]")
print("=" * 80)
print(f"\n[OK] Extracted 768-dimensional IndoBERT embeddings for {len(bert_features_df)} posts")
print(f"[OK] Saved to {output_path}")
print(f"[OK] File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")

print("\n[NEXT] Next steps:")
print("   1. Extract NIMA aesthetic features")
print("   2. Train model with new dataset (1,579 posts vs 348)")
print("   3. Expected: MAE < 120 (vs 135.21 previous)")
print("")

print("\n" + "=" * 80 + "\n")
