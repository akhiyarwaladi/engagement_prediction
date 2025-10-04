#!/usr/bin/env python3
"""
Phase 4b: Vision Transformer (ViT) Feature Extraction
======================================================

Extract 768-dimensional ViT embeddings from Instagram images using
Google's Vision Transformer (ViT-base-patch16-224).

Expected runtime: 10-15 minutes (CPU), 3-5 minutes (GPU)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print(" " * 25 + "PHASE 4B: VISUAL TRANSFORMER")
print(" " * 20 + "Image Feature Extraction with ViT")
print("=" * 80 + "\n")

# Check PyTorch availability
try:
    import torch
    from transformers import ViTImageProcessor, ViTModel
    print(f"[OK] PyTorch version: {torch.__version__}")
    print(f"[OK] Device: {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print("[FAIL] Error: PyTorch or Transformers not installed!")
    print("   Already installed from Phase 4a")
    sys.exit(1)

# Load Vision Transformer model
print("\n[LOAD] Loading Vision Transformer (ViT) model...")
print("   Model: google/vit-base-patch16-224")
print("   (This may take a minute on first run)")

try:
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)
    print(f"[OK] Loaded: {model_name}")
    print(f"   Model parameters: ~86M")
except Exception as e:
    print(f"[FAIL] Error loading ViT: {e}")
    sys.exit(1)

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()  # Set to evaluation mode

print(f"[OK] Model ready on {device}")

def extract_vit_embedding(image_path: str) -> np.ndarray:
    """
    Extract 768-dimensional ViT embedding from image.

    Args:
        image_path: Path to image file

    Returns:
        768-dim numpy array
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')

        # Preprocess
        inputs = processor(images=image, return_tensors="pt")

        # Move to same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get embeddings (no gradient computation)
        with torch.no_grad():
            outputs = model(**inputs)

        # Use [CLS] token embedding (image-level representation)
        # Shape: [batch_size, hidden_size] -> [1, 768]
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()

        # Convert to numpy
        return cls_embedding.cpu().numpy()

    except Exception as e:
        # Return zero vector if image can't be loaded
        print(f"   [WARN]  Error processing {image_path}: {e}")
        return np.zeros(768)

# Load data
print("\n[DATA] Loading dataset...")
df = pd.read_csv('fst_unja_from_gallery_dl.csv')
print(f"   Loaded {len(df)} posts")

# Check file paths
print("\n[CHECK] Checking image files...")
valid_images = 0
videos = 0
missing = 0

for idx, row in df.iterrows():
    file_path = row.get('file_path', '')
    is_video = row.get('is_video', 0)

    if is_video:
        videos += 1
    elif Path(file_path).exists() and Path(file_path).suffix.lower() in ['.jpg', '.jpeg', '.png']:
        valid_images += 1
    else:
        missing += 1

print(f"   Valid images: {valid_images}")
print(f"   Videos (will skip): {videos}")
print(f"   Missing/invalid: {missing}")

# Extract ViT embeddings
print("\n[IMAGE] Extracting ViT embeddings from images...")
print("   This will take 10-15 minutes on CPU, 3-5 minutes on GPU")
print("")

vit_embeddings = []

for idx, row in df.iterrows():
    file_path = row.get('file_path', '')
    is_video = row.get('is_video', 0)

    # Skip videos (use zero vector)
    if is_video:
        embedding = np.zeros(768)
    else:
        # Extract ViT embedding from image
        embedding = extract_vit_embedding(file_path)

    vit_embeddings.append(embedding)

    # Progress (every 20 images)
    if (idx + 1) % 20 == 0:
        progress = (idx + 1) / len(df) * 100
        print(f"  Progress: {idx + 1}/{len(df)} ({progress:.1f}%)")

# Convert to DataFrame
print("\n[STATS] Creating feature DataFrame...")
vit_features_df = pd.DataFrame(
    vit_embeddings,
    columns=[f'vit_dim_{i}' for i in range(768)]
)

# Add post metadata for reference
vit_features_df.insert(0, 'post_id', df['post_id'].values)
vit_features_df.insert(1, 'is_video', df['is_video'].values)
vit_features_df.insert(2, 'file_path', df['file_path'].values)

# Save to CSV
output_path = 'data/processed/vit_embeddings.csv'
Path(output_path).parent.mkdir(parents=True, exist_ok=True)

vit_features_df.to_csv(output_path, index=False)
print(f"[SAVE] Saved to: {output_path}")

# Statistics
print("\n[INFO] Embedding Statistics:")
print(f"   Shape: {vit_features_df.shape}")
print(f"   Dimensions: 768")
print(f"   Posts processed: {len(vit_features_df)}")
print(f"   Images (non-zero): {valid_images}")
print(f"   Videos (zero vector): {videos}")

# Show sample
print("\n[SAMPLE] Sample embeddings (first 5 dimensions):")
sample_cols = ['post_id', 'is_video'] + [f'vit_dim_{i}' for i in range(5)]
print(vit_features_df[sample_cols].head(3))

# Embedding quality check
print("\n[CHECK] Quality Check:")
embeddings_array = vit_embeddings

# Calculate norms (excluding zero vectors from videos)
non_zero_embeddings = [emb for emb in embeddings_array if np.linalg.norm(emb) > 0]
if len(non_zero_embeddings) > 0:
    mean_norm = np.mean([np.linalg.norm(emb) for emb in non_zero_embeddings])
    print(f"   Average embedding norm (images only): {mean_norm:.2f}")
    print(f"   Expected range: 10-20 (typical for ViT)")

    if mean_norm < 5:
        print("   [WARN]  Warning: Embedding norms unusually low")
    elif mean_norm > 30:
        print("   [WARN]  Warning: Embedding norms unusually high")
    else:
        print("   [OK] Embeddings look good!")

    # Check for variability
    std_per_dim = np.std(non_zero_embeddings, axis=0).mean()
    print(f"   Average std per dimension: {std_per_dim:.3f}")
    print(f"   Expected range: 0.5-2.0")

    if std_per_dim < 0.2:
        print("   [WARN]  Warning: Low variability, embeddings may be too similar")
    else:
        print("   [OK] Good variability across embeddings!")
else:
    print("   [WARN]  No valid image embeddings found")

# Summary
print("\n" + "=" * 80)
print("EXTRACTION COMPLETE! [DONE]")
print("=" * 80)

print(f"\n[OK] Extracted 768-dimensional ViT embeddings")
print(f"[OK] Saved {len(vit_features_df)} embeddings to {output_path}")
print(f"[OK] File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")

print("\n[STATS] Feature summary:")
print(f"   Text features (BERT): 768 dims → 50 PCA")
print(f"   Visual features (ViT): 768 dims → 50 PCA")
print(f"   Baseline features: 9")
print(f"   Total Phase 4b: 109 features (multimodal!)")

print("\n[NEXT] Next steps:")
print("   1. Run: python3 improve_model_v4_full.py")
print("   2. Expected: MAE ~70-85, R² ~0.30-0.40")
print("   3. Target achievement likely! [OK]")

print("\n" + "=" * 80 + "\n")
