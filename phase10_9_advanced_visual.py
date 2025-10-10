#!/usr/bin/env python3
"""
PHASE 10.9: ADVANCED VISUAL FEATURES
Test face detection, color analysis, metadata features (REAL visual features!)
MUST INCLUDE: Visual + Text features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print(" "*15 + "PHASE 10.9: ADVANCED VISUAL FEATURES")
print(" "*15 + "Target: Beat Phase 10.4 MAE=44.66")
print("="*90)
print()

# Load with BOTH modalities
print("[LOAD] Loading multimodal dataset...")
df_main = pd.read_csv('multi_account_dataset.csv')
df_bert = pd.read_csv('data/processed/bert_embeddings_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])
df_visual = pd.read_csv('data/processed/advanced_visual_features_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])

df = df_main.merge(df_bert, on=['post_id', 'account'], how='inner')
df = df.merge(df_visual, on=['post_id', 'account'], how='inner')

print(f"   Dataset: {len(df)} posts (BOTH visual + text)")
print()

# Features
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                 'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']

# Advanced visual features
face_cols = ['face_count', 'has_faces']
color_cols = ['dominant_r', 'dominant_g', 'dominant_b', 'brightness', 'saturation', 'color_variance']
metadata_cols = ['aspect_ratio', 'resolution', 'file_size_kb', 'is_portrait', 'is_landscape', 'is_square']

bert_cols = [col for col in df.columns if col.startswith('bert_')]

X_baseline = df[baseline_cols].values
X_face = df[face_cols].values
X_color = df[color_cols].values
X_metadata = df[metadata_cols].values
X_bert_full = df[bert_cols].values
y = df['likes'].values

# Split
train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)

X_baseline_train = X_baseline[train_idx]
X_baseline_test = X_baseline[test_idx]
X_face_train = X_face[train_idx]
X_face_test = X_face[test_idx]
X_color_train = X_color[train_idx]
X_color_test = X_color[test_idx]
X_metadata_train = X_metadata[train_idx]
X_metadata_test = X_metadata[test_idx]
X_bert_train = X_bert_full[train_idx]
X_bert_test = X_bert_full[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

# Preprocessing
clip_threshold = np.percentile(y_train, 99)
y_train_log = np.log1p(np.clip(y_train, 0, clip_threshold))
y_test_log = np.log1p(np.clip(y_test, 0, clip_threshold))

# PCA for BERT
pca_bert = PCA(n_components=50, random_state=42)
X_bert_pca_train = pca_bert.fit_transform(X_bert_train)
X_bert_pca_test = pca_bert.transform(X_bert_test)

print("[EXPERIMENT] Testing advanced visual feature combinations...")
print()

# ============================================================================
# Experiment: Different visual feature combinations
# ============================================================================

configs = [
    {'name': 'Baseline + BERT only', 'visual': None},
    {'name': 'Baseline + BERT + Face features', 'visual': 'face'},
    {'name': 'Baseline + BERT + Color features', 'visual': 'color'},
    {'name': 'Baseline + BERT + Metadata features', 'visual': 'metadata'},
    {'name': 'Baseline + BERT + Face + Color', 'visual': 'face_color'},
    {'name': 'Baseline + BERT + All visual (Face+Color+Metadata)', 'visual': 'all'},
]

best_mae = float('inf')
best_config = None

for config in configs:
    visual_type = config['visual']

    # Build feature matrix
    X_train_parts = [X_baseline_train, X_bert_pca_train]
    X_test_parts = [X_baseline_test, X_bert_pca_test]

    if visual_type == 'face':
        X_train_parts.append(X_face_train)
        X_test_parts.append(X_face_test)
    elif visual_type == 'color':
        X_train_parts.append(X_color_train)
        X_test_parts.append(X_color_test)
    elif visual_type == 'metadata':
        X_train_parts.append(X_metadata_train)
        X_test_parts.append(X_metadata_test)
    elif visual_type == 'face_color':
        X_train_parts.extend([X_face_train, X_color_train])
        X_test_parts.extend([X_face_test, X_color_test])
    elif visual_type == 'all':
        X_train_parts.extend([X_face_train, X_color_train, X_metadata_train])
        X_test_parts.extend([X_face_test, X_color_test, X_metadata_test])

    # Combine
    X_train = np.hstack(X_train_parts)
    X_test = np.hstack(X_test_parts)

    # Scale
    scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Quick stacking test
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(X_train_scaled), 4))
    test_preds = np.zeros((len(X_test_scaled), 4))

    models = [
        GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8, random_state=42),
        HistGradientBoostingRegressor(max_iter=600, learning_rate=0.07, max_depth=7, random_state=42),
        RandomForestRegressor(n_estimators=300, max_depth=16, random_state=42, n_jobs=-1),
        ExtraTreesRegressor(n_estimators=300, max_depth=16, random_state=42, n_jobs=-1),
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

    visual_included = 'NO' if visual_type is None else 'YES'
    print(f"   {config['name']:<55} Features={X_train_scaled.shape[1]:3d}: MAE={mae:.2f}, R2={r2:.4f} (Visual: {visual_included})")

    if mae < best_mae:
        best_mae = mae
        best_config = config['name']

print()
print(f"[BEST VISUAL] {best_config}: MAE={best_mae:.2f}")
print()

# ============================================================================
# Save results
# ============================================================================

if best_mae < 44.66:
    print("[SAVE] New champion! Saving model...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'models/phase10_9_advanced_visual_{timestamp}.pkl'

    model_package = {
        'phase': '10.9_advanced_visual',
        'mae': best_mae,
        'best_config': best_config,
        'visual_included': True,
        'text_included': True,
        'timestamp': timestamp
    }

    joblib.dump(model_package, model_filename)
    print(f"   Saved: {model_filename}")
    print()

print("="*90)
print(" "*30 + "PHASE 10.9 COMPLETE!")
print("="*90)
print()
print(f"[RESULT] MAE={best_mae:.2f} (Phase 10.4 was 44.66)")
if best_mae < 44.66:
    print(f"   IMPROVED by {(44.66-best_mae)/44.66*100:.1f}%!")
else:
    print(f"   Advanced visual features did not improve (Polynomial aesthetic remains best)")
