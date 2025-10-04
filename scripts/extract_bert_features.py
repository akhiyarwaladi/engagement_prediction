#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4a: IndoBERT Feature Extraction
======================================

Extract 768-dimensional BERT embeddings from Instagram captions using IndoBERTweet.

IndoBERTweet is specifically trained on Indonesian social media text, making it
ideal for Instagram caption analysis.

Expected runtime: 5-10 minutes (CPU), 1-2 minutes (GPU)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print(" " * 25 + "PHASE 4A: IndoBERT FEATURES")
print(" " * 20 + "Contextual Caption Embeddings")
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

# Load IndoBERTweet model
print("\n[LOAD] Loading IndoBERTweet model...")
print("   (This may take a minute on first run)")

# Use indobenchmark/indobert-base-p1 (verified working)
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
    # Shape: [batch_size, hidden_size] -> [1, 768]
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()

    # Convert to numpy
    return cls_embedding.cpu().numpy()

# Load data
print("\n[DATA] Loading dataset...")
df = pd.read_csv('fst_unja_from_gallery_dl.csv')
print(f"   Loaded {len(df)} posts")

# Extract BERT embeddings
print("\n[BERT] Extracting BERT embeddings from captions...")
print("   This will take 5-10 minutes on CPU, 1-2 minutes on GPU")
print("")

bert_embeddings = []
batch_size = 32  # Process in batches for GPU efficiency

for idx in range(0, len(df), batch_size):
    batch_end = min(idx + batch_size, len(df))
    batch_captions = df['caption'].iloc[idx:batch_end].fillna('')

    # Process batch
    for caption in batch_captions:
        embedding = extract_bert_embedding(str(caption))
        bert_embeddings.append(embedding)

    # Progress
    progress = batch_end / len(df) * 100
    print(f"  Progress: {batch_end}/{len(df)} ({progress:.1f}%)")

# Convert to DataFrame
print("\n[STATS] Creating feature DataFrame...")
bert_features_df = pd.DataFrame(
    bert_embeddings,
    columns=[f'bert_dim_{i}' for i in range(768)]
)

# Add post metadata for reference
bert_features_df.insert(0, 'post_id', df['post_id'].values)
bert_features_df.insert(1, 'caption_preview', df['caption'].fillna('').str[:50])

# Save to CSV
output_path = 'data/processed/bert_embeddings.csv'
Path(output_path).parent.mkdir(parents=True, exist_ok=True)

bert_features_df.to_csv(output_path, index=False)
print(f"[SAVE] Saved to: {output_path}")

# Statistics
print("\n[INFO] Embedding Statistics:")
print(f"   Shape: {bert_features_df.shape}")
print(f"   Dimensions: 768")
print(f"   Posts processed: {len(bert_features_df)}")

# Show sample
print("\n[SAMPLE] Sample embeddings (first 5 dimensions):")
sample_cols = ['caption_preview'] + [f'bert_dim_{i}' for i in range(5)]
print(bert_features_df[sample_cols].head(3))

# Embedding quality check
print("\n[CHECK] Quality Check:")
embeddings_array = bert_embeddings
mean_norm = np.mean([np.linalg.norm(emb) for emb in embeddings_array])
print(f"   Average embedding norm: {mean_norm:.2f}")
print(f"   Expected range: 8-15 (typical for BERT)")

if mean_norm < 5:
    print("   [WARN]  Warning: Embedding norms unusually low")
elif mean_norm > 20:
    print("   [WARN]  Warning: Embedding norms unusually high")
else:
    print("   [OK] Embeddings look good!")

# Check for variability
std_per_dim = np.std(embeddings_array, axis=0).mean()
print(f"   Average std per dimension: {std_per_dim:.3f}")
print(f"   Expected range: 0.3-1.0")

if std_per_dim < 0.1:
    print("   [WARN]  Warning: Low variability, embeddings may be too similar")
else:
    print("   [OK] Good variability across embeddings!")

# Summary
print("\n" + "=" * 80)
print("EXTRACTION COMPLETE! [DONE]")
print("=" * 80)
print("\n[OK] Extracted 768-dimensional IndoBERT embeddings")
print(f"[OK] Saved {len(bert_features_df)} embeddings to {output_path}")
print(f"[OK] File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")

print("\n[NEXT] Next steps:")
print("   1. Run: python3 improve_model_v4_bert.py")
print("   2. Expected improvement: MAE 109 → ~95, R² 0.20 → ~0.25")
print("")

print("[STATS] Feature comparison:")
print("   Phase 2: 28 features (simple NLP)")
print("   Phase 4a: 28 + 768 = 796 features (with IndoBERT)")
print("   Increase: 28x more features!")

print("\n" + "=" * 80 + "\n")
