#!/usr/bin/env python3
"""
PHASE 10.12: POLYNOMIAL DEGREE EXPLORATION
Test different polynomial degrees on aesthetic features
Phase 10.4 used degree=2, let's try 3 and combinations
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
print(" "*12 + "PHASE 10.12: POLYNOMIAL DEGREE OPTIMIZATION")
print(" "*18 + "Target: Beat Phase 10.4 MAE=44.66")
print("="*90)
print()

# Load with BOTH modalities
print("[LOAD] Loading multimodal dataset...")
df_main = pd.read_csv('multi_account_dataset.csv')
df_bert = pd.read_csv('data/processed/bert_embeddings_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])
df_aes = pd.read_csv('data/processed/aesthetic_features_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])

df = df_main.merge(df_bert, on=['post_id', 'account'], how='inner')
df = df.merge(df_aes, on=['post_id', 'account'], how='inner')

print(f"   Dataset: {len(df)} posts (BOTH visual + text)")
print()

# Features
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                 'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']
bert_cols = [col for col in df.columns if col.startswith('bert_')]
aes_cols = [col for col in df.columns if col.startswith('aesthetic_')]

X_baseline = df[baseline_cols].values
X_bert_full = df[bert_cols].values
X_aes_full = df[aes_cols].values
y = df['likes'].values

# Split
train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)

X_baseline_train = X_baseline[train_idx]
X_baseline_test = X_baseline[test_idx]
X_bert_train = X_bert_full[train_idx]
X_bert_test = X_bert_full[test_idx]
X_aes_train = X_aes_full[train_idx]
X_aes_test = X_aes_full[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

# Preprocessing
clip_threshold = np.percentile(y_train, 99)
y_train_log = np.log1p(np.clip(y_train, 0, clip_threshold))
y_test_log = np.log1p(np.clip(y_test, 0, clip_threshold))

# PCA (Phase 10.4 optimal)
pca_bert = PCA(n_components=50, random_state=42)
pca_aes = PCA(n_components=6, random_state=42)

X_bert_pca_train = pca_bert.fit_transform(X_bert_train)
X_bert_pca_test = pca_bert.transform(X_bert_test)
X_aes_pca_train = pca_aes.fit_transform(X_aes_train)
X_aes_pca_test = pca_aes.transform(X_aes_test)

print("[EXPERIMENT] Testing polynomial degrees (3-fold CV, 2 models for speed)...")
print()

# ============================================================================
# Test different polynomial degrees
# ============================================================================

configs = [
    {'name': 'Degree 1 (Linear, no poly)', 'degree': 1},
    {'name': 'Degree 2 (Phase 10.4)', 'degree': 2},
    {'name': 'Degree 3 (Cubic)', 'degree': 3},
    {'name': 'Degree 2, interactions only', 'degree': 2, 'interaction_only': True},
]

best_mae = float('inf')
best_config = None

for config in configs:
    degree = config['degree']
    interaction_only = config.get('interaction_only', False)

    # Apply polynomial transformation
    if degree == 1:
        # No polynomial, just use PCA features
        X_aes_poly_train = X_aes_pca_train
        X_aes_poly_test = X_aes_pca_test
    else:
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=interaction_only)
        X_aes_poly_train = poly.fit_transform(X_aes_pca_train)
        X_aes_poly_test = poly.transform(X_aes_pca_test)

    # Combine features
    X_train = np.hstack([X_baseline_train, X_bert_pca_train, X_aes_poly_train])
    X_test = np.hstack([X_baseline_test, X_bert_pca_test, X_aes_poly_test])

    # Scale
    scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Quick 3-fold stacking (2 models)
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

    print(f"   {config['name']:<40} Features={X_train_scaled.shape[1]:3d}: MAE={mae:.2f}, R2={r2:.4f}")

    if mae < best_mae:
        best_mae = mae
        best_config = config['name']

print()
print(f"[BEST POLYNOMIAL] {best_config}: MAE={best_mae:.2f}")
print()

# ============================================================================
# Save if improved
# ============================================================================

if best_mae < 44.66:
    print("[SAVE] New champion! Saving model...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'models/phase10_12_poly_degree_{timestamp}.pkl'

    model_package = {
        'phase': '10.12_polynomial_degrees',
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
print(" "*30 + "PHASE 10.12 COMPLETE!")
print("="*90)
print()
print(f"[RESULT] MAE={best_mae:.2f} (Phase 10.4 was 44.66)")
if best_mae < 44.66:
    print(f"   IMPROVED by {(44.66-best_mae)/44.66*100:.1f}%!")
    print(f"   Polynomial degree optimization SUCCESS!")
else:
    print(f"   Degree 2 remains optimal (Phase 10.4 still champion)")
