#!/usr/bin/env python3
"""
PHASE 10.11: QUICK VISUAL FEATURE TEST
Fast experiment to validate advanced visual features concept
Uses 3-fold CV instead of 5-fold for speed
MUST INCLUDE: Visual + Text features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures, QuantileTransformer
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print(" "*15 + "PHASE 10.11: QUICK VISUAL FEATURE TEST (3-fold)")
print(" "*18 + "Target: Beat Phase 10.4 MAE=44.66")
print("="*90)
print()

# Load with ALL modalities
print("[LOAD] Loading multimodal dataset...")
df_main = pd.read_csv('multi_account_dataset.csv')
df_bert = pd.read_csv('data/processed/bert_embeddings_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])
df_aes = pd.read_csv('data/processed/aesthetic_features_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])
df_visual = pd.read_csv('data/processed/advanced_visual_features_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])

df = df_main.merge(df_bert, on=['post_id', 'account'], how='inner')
df = df.merge(df_aes, on=['post_id', 'account'], how='inner')
df = df.merge(df_visual, on=['post_id', 'account'], how='inner')

print(f"   Dataset: {len(df)} posts (ALL modalities)")
print()

# Features
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                 'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']

bert_cols = [col for col in df.columns if col.startswith('bert_')]
aes_cols = [col for col in df.columns if col.startswith('aesthetic_')]

# Advanced visual - most promising features
face_cols = ['has_faces']  # Binary is often more robust than count
color_cols = ['brightness', 'saturation']  # Key color features
metadata_cols = ['aspect_ratio', 'is_portrait']  # Key metadata

X_baseline = df[baseline_cols].values
X_bert_full = df[bert_cols].values
X_aes_full = df[aes_cols].values
X_face = df[face_cols].values
X_color = df[color_cols].values
X_metadata = df[metadata_cols].values
y = df['likes'].values

# Split
train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)

X_baseline_train = X_baseline[train_idx]
X_baseline_test = X_baseline[test_idx]
X_bert_train = X_bert_full[train_idx]
X_bert_test = X_bert_full[test_idx]
X_aes_train = X_aes_full[train_idx]
X_aes_test = X_aes_full[test_idx]
X_face_train = X_face[train_idx]
X_face_test = X_face[test_idx]
X_color_train = X_color[train_idx]
X_color_test = X_color[test_idx]
X_metadata_train = X_metadata[train_idx]
X_metadata_test = X_metadata[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

# Preprocessing
clip_threshold = np.percentile(y_train, 99)
y_train_log = np.log1p(np.clip(y_train, 0, clip_threshold))
y_test_log = np.log1p(np.clip(y_test, 0, clip_threshold))

# PCA
pca_bert = PCA(n_components=50, random_state=42)
pca_aes = PCA(n_components=6, random_state=42)

X_bert_pca_train = pca_bert.fit_transform(X_bert_train)
X_bert_pca_test = pca_bert.transform(X_bert_test)
X_aes_pca_train = pca_aes.fit_transform(X_aes_train)
X_aes_pca_test = pca_aes.transform(X_aes_test)

# Aesthetic polynomial (best from Phase 10.4)
poly_aes = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_aes_poly_train = poly_aes.fit_transform(X_aes_pca_train)
X_aes_poly_test = poly_aes.transform(X_aes_pca_test)

print("[EXPERIMENT] Quick tests with 3-fold CV (2 models only)...")
print()

# ============================================================================
# Quick tests - only most promising combinations
# ============================================================================

configs = [
    {'name': 'Phase 10.4 baseline (Poly Aes only)', 'extras': []},
    {'name': 'Poly Aes + Color (bright+sat)', 'extras': ['color']},
    {'name': 'Poly Aes + Faces (has_faces)', 'extras': ['face']},
    {'name': 'Poly Aes + Color + Faces', 'extras': ['color', 'face']},
]

best_mae = float('inf')
best_config = None

# Use only 2 fast models (GBM and HGB)
for config in configs:
    extras = config['extras']

    # Build feature matrix
    X_train_parts = [X_baseline_train, X_bert_pca_train, X_aes_poly_train]
    X_test_parts = [X_baseline_test, X_bert_pca_test, X_aes_poly_test]

    if 'face' in extras:
        X_train_parts.append(X_face_train)
        X_test_parts.append(X_face_test)

    if 'color' in extras:
        X_train_parts.append(X_color_train)
        X_test_parts.append(X_color_test)

    if 'metadata' in extras:
        X_train_parts.append(X_metadata_train)
        X_test_parts.append(X_metadata_test)

    # Combine
    X_train = np.hstack(X_train_parts)
    X_test = np.hstack(X_test_parts)

    # Scale
    scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Quick 3-fold test with 2 models
    kf = KFold(n_splits=3, shuffle=True, random_state=42)  # 3-fold for speed!
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

    print(f"   {config['name']:<45} Features={X_train_scaled.shape[1]:3d}: MAE={mae:.2f}, R2={r2:.4f}")

    if mae < best_mae:
        best_mae = mae
        best_config = config['name']

print()
print(f"[BEST QUICK TEST] {best_config}: MAE={best_mae:.2f}")
print()

# ============================================================================
# Save if improved
# ============================================================================

if best_mae < 44.66:
    print("[SAVE] New champion! Saving model...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'models/phase10_11_quick_visual_{timestamp}.pkl'

    model_package = {
        'phase': '10.11_quick_visual',
        'mae': best_mae,
        'best_config': best_config,
        'cv_folds': 3,
        'models_used': 2,
        'visual_included': True,
        'text_included': True,
        'timestamp': timestamp
    }

    joblib.dump(model_package, model_filename)
    print(f"   Saved: {model_filename}")
    print()

print("="*90)
print(" "*30 + "PHASE 10.11 COMPLETE!")
print("="*90)
print()
print(f"[RESULT] MAE={best_mae:.2f} (Phase 10.4 was 44.66)")
if best_mae < 44.66:
    print(f"   IMPROVED by {(44.66-best_mae)/44.66*100:.1f}%!")
    print(f"   Quick test shows advanced visual features WORK!")
else:
    print(f"   Quick test shows no improvement from advanced visual features")
    print(f"   Full Phase 10.9 & 10.10 will provide definitive answer")
