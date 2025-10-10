#!/usr/bin/env python3
"""
PHASE 10.15: SMART VISUAL OPTIMIZATION
Target: Beat Phase 10.9 (MAE=44.05) with intelligent visual feature combinations
FAST: 3-fold CV, 2 models only
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print(" "*10 + "PHASE 10.15: SMART VISUAL OPTIMIZATION (FAST!)")
print(" "*15 + "Target: Beat Phase 10.9 MAE=44.05")
print("="*90)
print()

# Load data
print("[LOAD] Loading multimodal dataset...")
df_main = pd.read_csv('multi_account_dataset.csv')
df_bert = pd.read_csv('data/processed/bert_embeddings_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])
df_visual = pd.read_csv('data/processed/advanced_visual_features_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])

df = df_main.merge(df_bert, on=['post_id', 'account'], how='inner')
df = df.merge(df_visual, on=['post_id', 'account'], how='inner')

print(f"   Dataset: {len(df)} posts")
print()

# Baseline
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                 'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']

# BERT
bert_cols = [col for col in df.columns if col.startswith('bert_')]

# Visual features
metadata_base = ['aspect_ratio', 'resolution', 'file_size_kb', 'is_portrait', 'is_landscape', 'is_square']
color_features = ['brightness', 'saturation', 'dominant_r', 'dominant_g', 'dominant_b', 'color_variance']
face_features = ['face_count', 'has_faces']

# Smart feature engineering (minimal, targeted)
print("[ENGINEER] Creating smart visual features...")

# Only useful interactions (based on Phase 10.13 learnings)
df['aspect_x_video'] = df['aspect_ratio'] * df['is_video']
df['resolution_norm'] = np.log1p(df['resolution'])  # Log scale for resolution
df['brightness_norm'] = df['brightness'] / 255.0  # Normalize brightness

# Categorical orientation (better than binary flags)
df['orientation_cat'] = 0  # default square
df.loc[df['is_portrait'] == 1, 'orientation_cat'] = 1
df.loc[df['is_landscape'] == 1, 'orientation_cat'] = 2

smart_features = ['aspect_x_video', 'resolution_norm', 'brightness_norm', 'orientation_cat']

print(f"   Created 4 smart features!")
print()

# Test configurations
configs = [
    {
        'name': 'Phase 10.9 Baseline (Metadata only)',
        'features': metadata_base
    },
    {
        'name': 'Metadata + Smart Engineering',
        'features': metadata_base + smart_features
    },
    {
        'name': 'Metadata + Brightness (selective color)',
        'features': metadata_base + ['brightness', 'saturation']
    },
    {
        'name': 'Metadata + Smart + Brightness',
        'features': metadata_base + smart_features + ['brightness', 'saturation']
    },
]

results = []

for config in configs:
    print(f"[TEST] {config['name']}")

    # Combine features
    visual_features = config['features']

    X_baseline = df[baseline_cols].values
    X_bert_full = df[bert_cols].values
    X_visual = df[visual_features].values
    y = df['likes'].values

    # Split
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)

    X_baseline_train = X_baseline[train_idx]
    X_baseline_test = X_baseline[test_idx]
    X_bert_train = X_bert_full[train_idx]
    X_bert_test = X_bert_full[test_idx]
    X_visual_train = X_visual[train_idx]
    X_visual_test = X_visual[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # Preprocessing
    clip_threshold = np.percentile(y_train, 99)
    y_train_log = np.log1p(np.clip(y_train, 0, clip_threshold))
    y_test_log = np.log1p(np.clip(y_test, 0, clip_threshold))

    # PCA BERT
    pca_bert = PCA(n_components=50, random_state=42)
    X_bert_pca_train = pca_bert.fit_transform(X_bert_train)
    X_bert_pca_test = pca_bert.transform(X_bert_test)

    # Combine
    X_train = np.hstack([X_baseline_train, X_bert_pca_train, X_visual_train])
    X_test = np.hstack([X_baseline_test, X_bert_pca_test, X_visual_test])

    # Scale
    scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Quick stacking (3-fold, 2 models for speed)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(X_train_scaled), 2))
    test_preds = np.zeros((len(X_test_scaled), 2))

    models = [
        GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8, random_state=42),
        HistGradientBoostingRegressor(max_iter=600, learning_rate=0.07, max_depth=7, random_state=42),
    ]

    for i, model in enumerate(models):
        # OOF
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
            m = model.__class__(**model.get_params())
            m.fit(X_train_scaled[tr_idx], y_train_log[tr_idx])
            oof_preds[val_idx, i] = m.predict(X_train_scaled[val_idx])

        # Test
        model.fit(X_train_scaled, y_train_log)
        test_preds[:, i] = model.predict(X_test_scaled)

    # Meta
    meta = Ridge(alpha=10)
    meta.fit(oof_preds, y_train_log)

    y_pred_log = meta.predict(test_preds)
    y_pred = np.expm1(y_pred_log)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        'config': config['name'],
        'features': len(visual_features),
        'total_features': X_train.shape[1],
        'mae': mae,
        'r2': r2
    })

    print(f"   MAE={mae:.2f}, R2={r2:.4f} ({len(visual_features)} visual features, {X_train.shape[1]} total)")
    print()

# Summary
print("="*90)
print(" "*25 + "PHASE 10.15 RESULTS")
print("="*90)
print()

for i, res in enumerate(results, 1):
    status = "[WINNER]" if res['mae'] < 44.05 else "        "
    print(f"{status} {i}. {res['config']}")
    print(f"         MAE={res['mae']:.2f}, R2={res['r2']:.4f} ({res['features']} visual, {res['total_features']} total)")
    print()

# Find best
best = min(results, key=lambda x: x['mae'])
print("="*90)
print(f"[CHAMPION] {best['config']}")
print(f"   MAE={best['mae']:.2f}, R2={best['r2']:.4f}")
if best['mae'] < 44.05:
    improvement = (44.05 - best['mae']) / 44.05 * 100
    print(f"   [IMPROVED] Beat Phase 10.9 by {improvement:.2f}%!")
else:
    print(f"   Phase 10.9 remains champion (MAE=44.05)")
print("="*90)
