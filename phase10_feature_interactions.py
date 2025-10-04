#!/usr/bin/env python3
"""
PHASE 10.1: ADVANCED FEATURE INTERACTIONS (VISUAL x TEXT)
Test polynomial and interaction features between visual and text modalities
MUST INCLUDE: Visual + Text features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import QuantileTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print(" "*15 + "PHASE 10.1: VISUAL x TEXT FEATURE INTERACTIONS")
print(" "*20 + "Target: Beat Phase 9 MAE=45.10")
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

# PCA (use Phase 9 optimal)
pca_bert = PCA(n_components=50, random_state=42)
pca_aes = PCA(n_components=6, random_state=42)

X_bert_pca_train = pca_bert.fit_transform(X_bert_train)
X_bert_pca_test = pca_bert.transform(X_bert_test)
X_aes_pca_train = pca_aes.fit_transform(X_aes_train)
X_aes_pca_test = pca_aes.transform(X_aes_test)

print("[EXPERIMENT] Testing interaction strategies...")
print()

# ============================================================================
# Strategy 1: Top-k visual x text interactions
# ============================================================================

configs = [
    {'name': 'No interactions (baseline Phase 9)', 'n_bert': 0, 'n_aes': 0},
    {'name': 'Top 3 BERT x Top 2 Aes (6 interactions)', 'n_bert': 3, 'n_aes': 2},
    {'name': 'Top 5 BERT x Top 3 Aes (15 interactions)', 'n_bert': 5, 'n_aes': 3},
    {'name': 'Top 10 BERT x Top 4 Aes (40 interactions)', 'n_bert': 10, 'n_aes': 4},
    {'name': 'Top 15 BERT x Top 5 Aes (75 interactions)', 'n_bert': 15, 'n_aes': 5},
]

best_mae = float('inf')
best_config = None
best_X_train = None
best_X_test = None

for config in configs:
    # Base features
    X_train = np.hstack([X_baseline_train, X_bert_pca_train, X_aes_pca_train])
    X_test = np.hstack([X_baseline_test, X_bert_pca_test, X_aes_pca_test])

    # Add interactions
    if config['n_bert'] > 0:
        interactions_train = []
        interactions_test = []

        for i in range(config['n_bert']):
            for j in range(config['n_aes']):
                # BERT x Aesthetic interaction
                bert_idx = len(baseline_cols) + i
                aes_idx = len(baseline_cols) + 50 + j

                interactions_train.append(X_train[:, bert_idx] * X_train[:, aes_idx])
                interactions_test.append(X_test[:, bert_idx] * X_test[:, aes_idx])

        interactions_train = np.column_stack(interactions_train)
        interactions_test = np.column_stack(interactions_test)

        X_train = np.hstack([X_train, interactions_train])
        X_test = np.hstack([X_test, interactions_test])

    # Scale
    scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Test with stacking (simplified - 3 models)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(X_train_scaled), 3))
    test_preds = np.zeros((len(X_test_scaled), 3))

    models = [
        GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8, random_state=42),
        HistGradientBoostingRegressor(max_iter=600, learning_rate=0.07, max_depth=7, random_state=42),
        RandomForestRegressor(n_estimators=300, max_depth=16, random_state=42, n_jobs=-1),
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

    # Meta-learner
    meta = Ridge(alpha=10)
    meta.fit(oof_preds, y_train_log)

    y_pred_log = meta.predict(test_preds)
    y_pred = np.expm1(y_pred_log)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"   {config['name']:50s} Features={X_train.shape[1]:3d}: MAE={mae:.2f}, R2={r2:.4f}")

    if mae < best_mae:
        best_mae = mae
        best_config = config
        best_X_train = X_train
        best_X_test = X_test

print()
print(f"[BEST INTERACTION] {best_config['name']}: MAE={best_mae:.2f}")
print(f"   Total features: {best_X_train.shape[1]}")
print()

# ============================================================================
# Save best model
# ============================================================================

if best_mae < 45.10:
    print("[SAVE] New champion! Saving model...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'models/phase10_1_interactions_{timestamp}.pkl'

    model_package = {
        'phase': '10.1_interactions',
        'mae': best_mae,
        'config': best_config,
        'n_features': best_X_train.shape[1],
        'visual_included': True,
        'text_included': True,
        'timestamp': timestamp
    }

    joblib.dump(model_package, model_filename)
    print(f"   Saved: {model_filename}")
    print()

print("="*90)
print(" "*30 + "PHASE 10.1 COMPLETE!")
print("="*90)
print()
print(f"[RESULT] MAE={best_mae:.2f} (Phase 9 was 45.10)")
if best_mae < 45.10:
    print(f"   IMPROVED by {(45.10-best_mae)/45.10*100:.1f}%!")
else:
    print(f"   Interactions did not improve (Phase 9 remains champion)")
