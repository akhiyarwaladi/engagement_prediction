#!/usr/bin/env python3
"""
Extract Aesthetic Quality Features for Instagram Posts
Based on research: NIMA, composition analysis, saliency features
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

    Technical Quality:
    - Blur/sharpness (Laplacian variance)
    - Noise level (local variance)
    - Exposure (brightness distribution)

    Aesthetic Quality:
    - Color harmony (HSV color distribution)
    - Color saturation appeal
    - Luminance contrast
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return {}

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
    # Color harmony: std of hue values (lower = more harmonious)
    hue_std = np.std(hsv[:, :, 0])

    # Saturation appeal: high saturation is more appealing
    saturation_mean = np.mean(hsv[:, :, 1])
    saturation_std = np.std(hsv[:, :, 1])

    # Luminance contrast
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

def extract_composition_features(image_path):
    """
    Extract composition analysis features

    - Rule of thirds alignment
    - Visual balance (weight distribution)
    - Symmetry score
    - Edge density (complexity)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return {}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Edge detection for composition analysis
    edges = cv2.Canny(gray, 100, 200)

    # Rule of thirds: divide into 9 regions
    h_third = h // 3
    w_third = w // 3

    regions = [
        edges[0:h_third, 0:w_third],           # top-left
        edges[0:h_third, w_third:2*w_third],   # top-center
        edges[0:h_third, 2*w_third:w],         # top-right
        edges[h_third:2*h_third, 0:w_third],   # mid-left
        edges[h_third:2*h_third, w_third:2*w_third],  # center
        edges[h_third:2*h_third, 2*w_third:w],  # mid-right
        edges[2*h_third:h, 0:w_third],         # bottom-left
        edges[2*h_third:h, w_third:2*w_third], # bottom-center
        edges[2*h_third:h, 2*w_third:w]        # bottom-right
    ]

    # Edge density per region
    region_densities = [np.sum(r) / r.size for r in regions]

    # Rule of thirds score: edges on intersection points (center region)
    rule_of_thirds_score = region_densities[4]  # center region density

    # Visual balance: std of region densities (lower = more balanced)
    visual_balance = np.std(region_densities)

    # Symmetry: compare left vs right halves
    left_half = edges[:, :w//2]
    right_half = cv2.flip(edges[:, w//2:], 1)  # flip horizontally
    min_width = min(left_half.shape[1], right_half.shape[1])
    symmetry_score = np.sum(left_half[:, :min_width] == right_half[:, :min_width]) / (h * min_width)

    # Edge density (complexity)
    edge_density = np.sum(edges) / edges.size

    return {
        'composition_rule_of_thirds': float(rule_of_thirds_score),
        'composition_balance': float(visual_balance),
        'composition_symmetry': float(symmetry_score),
        'composition_edge_density': float(edge_density)
    }

def extract_saliency_features(image_path):
    """
    Extract saliency (attention) features

    - Center bias: content concentration in center
    - Attention spread: how distributed is visual attention
    - Subject isolation: foreground vs background separation
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return {}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Simple saliency: use edge + color variance
    edges = cv2.Canny(gray, 100, 200)

    # Center bias: compare center region vs periphery
    center_h = slice(h//4, 3*h//4)
    center_w = slice(w//4, 3*w//4)
    center_edges = edges[center_h, center_w]
    periphery_edges = edges.copy()
    periphery_edges[center_h, center_w] = 0

    center_density = np.sum(center_edges) / center_edges.size
    periphery_density = np.sum(periphery_edges) / periphery_edges.size
    center_bias = center_density / (periphery_density + 1e-6)

    # Attention spread: std of edge density across image
    patch_size = 32
    edge_patches = []
    for i in range(0, h-patch_size, patch_size):
        for j in range(0, w-patch_size, patch_size):
            patch = edges[i:i+patch_size, j:j+patch_size]
            edge_patches.append(np.sum(patch) / patch.size)

    attention_spread = np.std(edge_patches) if edge_patches else 0.0

    # Subject isolation: use color variance
    # High variance in center, low in periphery = good isolation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    center_hsv = hsv[center_h, center_w]
    periphery_hsv = hsv.copy()
    periphery_hsv[center_h, center_w] = 0

    center_color_var = np.var(center_hsv)
    periphery_color_var = np.var(periphery_hsv[periphery_hsv != 0]) if np.any(periphery_hsv != 0) else 1.0
    subject_isolation = center_color_var / (periphery_color_var + 1e-6)

    return {
        'saliency_center_bias': float(center_bias),
        'saliency_attention_spread': float(attention_spread),
        'saliency_subject_isolation': float(subject_isolation)
    }

def extract_color_appeal_features(image_path):
    """
    Extract color appeal features based on color theory

    - Vibrancy: overall color vividness
    - Warmth/coolness: color temperature
    - Color diversity: number of distinct colors
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return {}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Vibrancy: high saturation + high value
    vibrancy = np.mean(s * v / 255.0)

    # Color temperature: warm (red/orange/yellow) vs cool (blue/green)
    # Hue in OpenCV: 0-180 (0=red, 60=yellow, 120=green, 150=blue)
    warm_mask = ((h < 30) | (h > 150)).astype(float)  # red, orange
    cool_mask = ((h >= 60) & (h <= 120)).astype(float)  # green, blue
    warmth_score = np.sum(warm_mask * s) / np.sum(s + 1e-6)

    # Color diversity: quantize colors and count unique
    h_quantized = (h // 30) * 30  # 6 hue bins
    s_quantized = (s // 64) * 64  # 4 saturation bins
    colors = h_quantized * 1000 + s_quantized
    unique_colors = len(np.unique(colors))
    color_diversity = unique_colors / 24.0  # normalize (6 hue * 4 sat = 24 max)

    return {
        'color_vibrancy': float(vibrancy),
        'color_warmth': float(warmth_score),
        'color_diversity': float(color_diversity)
    }

def main():
    print("\n" + "="*80)
    print(" "*20 + "AESTHETIC QUALITY FEATURE EXTRACTION")
    print(" "*15 + "NIMA-inspired + Composition + Saliency Features")
    print("="*80)

    # Load baseline dataset
    baseline_df = pd.read_csv('data/processed/baseline_dataset.csv')
    print(f"\n[DATA] Loaded {len(baseline_df)} posts")

    # Find image files
    gallery_path = Path('gallery-dl/instagram/fst_unja')

    aesthetic_features_list = []

    print("\n[EXTRACT] Processing images...")
    for idx, row in tqdm(baseline_df.iterrows(), total=len(baseline_df)):
        post_id = row['post_id']
        is_video = row['is_video']

        # Find image file
        image_files = list(gallery_path.glob(f"{post_id}.*"))
        image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

        if not image_files:
            # Videos: use thumbnail or skip
            if is_video:
                # Try to find thumbnail or use zero features
                aesthetic_features = {
                    'post_id': post_id,
                    'aesthetic_sharpness': 0.0,
                    'aesthetic_noise': 0.0,
                    'aesthetic_brightness': 0.0,
                    'aesthetic_exposure_quality': 0.0,
                    'aesthetic_color_harmony': 0.0,
                    'aesthetic_saturation': 0.0,
                    'aesthetic_saturation_variance': 0.0,
                    'aesthetic_luminance_contrast': 0.0,
                    'composition_rule_of_thirds': 0.0,
                    'composition_balance': 0.0,
                    'composition_symmetry': 0.0,
                    'composition_edge_density': 0.0,
                    'saliency_center_bias': 0.0,
                    'saliency_attention_spread': 0.0,
                    'saliency_subject_isolation': 0.0,
                    'color_vibrancy': 0.0,
                    'color_warmth': 0.0,
                    'color_diversity': 0.0
                }
            else:
                print(f"[WARN] No image found for {post_id}")
                continue
        else:
            image_path = image_files[0]

            # Extract all features
            nima_features = extract_nima_style_features(image_path)
            composition_features = extract_composition_features(image_path)
            saliency_features = extract_saliency_features(image_path)
            color_features = extract_color_appeal_features(image_path)

            aesthetic_features = {
                'post_id': post_id,
                **nima_features,
                **composition_features,
                **saliency_features,
                **color_features
            }

        aesthetic_features_list.append(aesthetic_features)

    # Create DataFrame
    aesthetic_df = pd.DataFrame(aesthetic_features_list)

    # Save
    output_path = 'data/processed/aesthetic_features.csv'
    aesthetic_df.to_csv(output_path, index=False)

    print(f"\n[SAVE] Aesthetic features saved: {output_path}")
    print(f"       Total posts: {len(aesthetic_df)}")
    print(f"       Total features: {len(aesthetic_df.columns) - 1}")

    # Show feature summary
    print("\n[SUMMARY] Feature Statistics:")
    print("-" * 80)
    for col in aesthetic_df.columns:
        if col == 'post_id':
            continue
        values = aesthetic_df[col]
        print(f"  {col:35} mean={values.mean():8.3f} std={values.std():8.3f} "
              f"min={values.min():8.3f} max={values.max():8.3f}")

    print("\n" + "="*80)
    print("AESTHETIC FEATURE EXTRACTION COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
