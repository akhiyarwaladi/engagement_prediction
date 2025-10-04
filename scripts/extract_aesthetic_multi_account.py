#!/usr/bin/env python3
"""
Extract NIMA Aesthetic Features for Multi-Account Dataset
Extract 8 aesthetic quality features for 8,610 posts (8 UNJA accounts)
"""

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def extract_nima_style_features(image_path):
    """
    Extract NIMA-inspired aesthetic and technical quality features

    8 Features:
    - Sharpness (Laplacian variance)
    - Noise level (local variance std)
    - Brightness (mean pixel intensity)
    - Exposure quality (brightness std)
    - Color harmony (hue std - lower is better)
    - Saturation (mean saturation)
    - Saturation variance (saturation std)
    - Luminance contrast (brightness_std/brightness)
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Technical quality metrics
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Noise level: std of local variance
        local_vars = []
        h, w = gray.shape
        for i in range(0, h-20, 20):
            for j in range(0, w-20, 20):
                patch = gray[i:i+20, j:j+20]
                local_vars.append(np.var(patch))
        noise_level = np.std(local_vars) if local_vars else 0.0

        # Exposure quality
        brightness = np.mean(gray)
        brightness_std = np.std(gray)

        # Aesthetic quality metrics
        hue_std = np.std(hsv[:, :, 0])
        saturation_mean = np.mean(hsv[:, :, 1])
        saturation_std = np.std(hsv[:, :, 1])
        luminance_contrast = brightness_std / (brightness + 1e-6)

        return {
            'aesthetic_sharpness': float(sharpness),
            'aesthetic_noise': float(noise_level),
            'aesthetic_brightness': float(brightness),
            'aesthetic_exposure_quality': float(brightness_std),
            'aesthetic_color_harmony': float(hue_std),
            'aesthetic_saturation': float(saturation_mean),
            'aesthetic_saturation_variance': float(saturation_std),
            'aesthetic_luminance_contrast': float(luminance_contrast)
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

print("\n" + "=" * 80)
print(" " * 20 + "MULTI-ACCOUNT NIMA EXTRACTION")
print(" " * 15 + "Aesthetic Features for 8,610 Posts")
print("=" * 80 + "\n")

# Load multi-account dataset
print("[DATA] Loading multi-account dataset...")
df = pd.read_csv('multi_account_dataset.csv')
print(f"   Loaded {len(df)} posts from 8 UNJA accounts")
print(f"   fst_unja: {len(df[df['account'] == 'fst_unja'])} posts")
print(f"   univ.jambi: {len(df[df['account'] == 'univ.jambi'])} posts")
print(f"   Other 6 accounts: {len(df) - len(df[df['account'].isin(['fst_unja', 'univ.jambi'])])} posts")

# Extract NIMA features
print("\n[NIMA] Extracting aesthetic features from images...")
print("   This will take ~20-30 minutes for 8,610 images")
print("")

results = []
errors = 0

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
    account = row['account']
    post_id = row['post_id']

    # Find image file
    # Try different extensions
    image_dir = Path(f'gallery-dl/instagram/{account}')
    image_path = None

    # Try different formats
    for ext in ['.jpg', '.png', '.webp', '.jpeg']:
        potential_path = image_dir / f"{post_id}{ext}"
        if potential_path.exists():
            image_path = potential_path
            break

    # If still not found, try with underscore pattern (carousel images)
    if image_path is None:
        # Search for files starting with post_id
        matches = list(image_dir.glob(f"{post_id}_*"))
        if matches:
            # Use first match (main image)
            image_path = matches[0]

    # Extract features
    if image_path and image_path.exists():
        features = extract_nima_style_features(image_path)
        if features:
            features['post_id'] = post_id
            features['account'] = account
            results.append(features)
        else:
            errors += 1
    else:
        # Image not found - use zeros
        results.append({
            'post_id': post_id,
            'account': account,
            'aesthetic_sharpness': 0.0,
            'aesthetic_noise': 0.0,
            'aesthetic_brightness': 0.0,
            'aesthetic_exposure_quality': 0.0,
            'aesthetic_color_harmony': 0.0,
            'aesthetic_saturation': 0.0,
            'aesthetic_saturation_variance': 0.0,
            'aesthetic_luminance_contrast': 0.0
        })
        errors += 1

# Create DataFrame
print("\n[STATS] Creating feature DataFrame...")
aesthetic_df = pd.DataFrame(results)

# Reorder columns
cols = ['post_id', 'account'] + [c for c in aesthetic_df.columns if c not in ['post_id', 'account']]
aesthetic_df = aesthetic_df[cols]

# Save to CSV
output_path = 'data/processed/aesthetic_features_multi_account.csv'
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
aesthetic_df.to_csv(output_path, index=False)

print(f"[SAVE] Saved to: {output_path}")

# Statistics
print("\n[INFO] Extraction Statistics:")
print(f"   Total posts: {len(aesthetic_df)}")
print(f"   Successful: {len(aesthetic_df) - errors}")
print(f"   Errors/Missing: {errors}")
print(f"   Success rate: {(len(aesthetic_df) - errors) / len(aesthetic_df) * 100:.1f}%")

# Show sample
print("\n[SAMPLE] Sample features:")
print(aesthetic_df.head(3))

# Feature statistics
print("\n[CHECK] Feature Statistics:")
feature_cols = [c for c in aesthetic_df.columns if c.startswith('aesthetic_')]
for col in feature_cols:
    print(f"   {col:35} mean={aesthetic_df[col].mean():.2f}, std={aesthetic_df[col].std():.2f}")

print("\n" + "=" * 80)
print("AESTHETIC EXTRACTION COMPLETE!")
print("=" * 80)
print(f"\n[OK] Extracted 8 NIMA aesthetic features for {len(aesthetic_df)} posts")
print(f"[OK] Saved to {output_path}")
print(f"[OK] File size: {Path(output_path).stat().st_size / 1024:.2f} KB")

print("\n[NEXT] Next steps:")
print("   1. Train model with expanded dataset (8,610 posts)")
print("   2. Compare: 1,949 vs 8,610 posts performance")
print("   3. Expected: MAE < 60 (vs 94.54 with 1,949 posts)")
print("")

print("\n" + "=" * 80 + "\n")
