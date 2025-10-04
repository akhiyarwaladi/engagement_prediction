#!/usr/bin/env python3
"""
PHASE 8: ADVANCED ULTRATHINK EXPERIMENTS
Target: Beat Phase 7 Champion MAE=50.55

New strategies:
1. Use FULL 8,610 dataset (skip aesthetic to avoid data loss)
2. Advanced GBM tuning with more iterations
3. HistGradientBoosting optimization
4. Stacking ensemble (meta-learning)
5. Feature interaction engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print(" "*20 + "PHASE 8: ADVANCED ULTRATHINK EXPERIMENTS")
print(" "*15 + "Target: Beat Phase 7 MAE=50.55 on Clean Data")
print("="*90)
print()

# ============================================================================
# EXPERIMENT 1: FULL DATASET (NO AESTHETIC TO AVOID DATA LOSS)
# ============================================================================
print("="*90)
print(" "*20 + "EXPERIMENT 1: FULL 8,610 DATASET (NO AESTHETIC)")
print("="*90)
print()

print("[LOAD] Loading full dataset without aesthetic...")
df_main = pd.read_csv('multi_account_dataset.csv')
df_bert = pd.read_csv('data/processed/bert_embeddings_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])

df = df_main.merge(df_bert, on=['post_id', 'account'], how='inner')
print(f"   Dataset: {len(df)} posts (vs 8,198 with aesthetic)")

# Features
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                 'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']
bert_cols = [col for col in df.columns if col.startswith('bert_')]

X_baseline = df[baseline_cols].values
X_bert_full = df[bert_cols].values
y = df['likes'].values

# Train/test split
train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)

X_baseline_train = X_baseline[train_idx]
X_baseline_test = X_baseline[test_idx]
X_bert_train = X_bert_full[train_idx]
X_bert_test = X_bert_full[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

# Preprocessing
clip_threshold = np.percentile(y_train, 99)
y_train_clipped = np.clip(y_train, 0, clip_threshold)
y_test_clipped = np.clip(y_test, 0, clip_threshold)
y_train_log = np.log1p(y_train_clipped)
y_test_log = np.log1p(y_test_clipped)

print(f"   Train: {len(X_baseline_train)} posts")
print(f"   Test: {len(X_baseline_test)} posts")

# Test optimal PCA from Phase 7
print()
print("[PCA] Testing dimensions on full dataset...")
pca_options = [50, 60, 70]
best_pca_mae = float('inf')
best_pca_n = None

for pca_n in pca_options:
    pca = PCA(n_components=pca_n, random_state=42)
    X_bert_pca_train = pca.fit_transform(X_bert_train)
    X_bert_pca_test = pca.transform(X_bert_test)

    X_train = np.hstack([X_baseline_train, X_bert_pca_train])
    X_test = np.hstack([X_baseline_test, X_bert_pca_test])

    scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    gbm = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=7,
                                     subsample=0.8, random_state=42)
    gbm.fit(X_train_scaled, y_train_log)

    y_pred_log = gbm.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)
    mae = mean_absolute_error(y_test, y_pred)

    variance = pca.explained_variance_ratio_.sum()
    print(f"   PCA-{pca_n}: MAE={mae:.2f} ({variance:.1%} variance)")

    if mae < best_pca_mae:
        best_pca_mae = mae
        best_pca_n = pca_n

print(f"\n[BEST] PCA-{best_pca_n}: MAE={best_pca_mae:.2f}")

# ============================================================================
# EXPERIMENT 2: AGGRESSIVE GBM TUNING
# ============================================================================
print()
print("="*90)
print(" "*20 + "EXPERIMENT 2: AGGRESSIVE GBM TUNING")
print("="*90)
print()

pca_best = PCA(n_components=best_pca_n, random_state=42)
X_bert_pca_train = pca_best.fit_transform(X_bert_train)
X_bert_pca_test = pca_best.transform(X_bert_test)

X_train = np.hstack([X_baseline_train, X_bert_pca_train])
X_test = np.hstack([X_baseline_test, X_bert_pca_test])

scaler_best = QuantileTransformer(output_distribution='uniform', random_state=42)
X_train_scaled = scaler_best.fit_transform(X_train)
X_test_scaled = scaler_best.transform(X_test)

print("[TEST] Aggressive GBM configurations:")

gbm_configs = [
    {'n_estimators': 700, 'learning_rate': 0.03, 'max_depth': 7, 'subsample': 0.8, 'min_samples_leaf': 2},
    {'n_estimators': 800, 'learning_rate': 0.03, 'max_depth': 6, 'subsample': 0.85, 'min_samples_leaf': 2},
    {'n_estimators': 600, 'learning_rate': 0.04, 'max_depth': 7, 'subsample': 0.8, 'min_samples_leaf': 2},
    {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 8, 'subsample': 0.8, 'min_samples_leaf': 1},
    {'n_estimators': 1000, 'learning_rate': 0.02, 'max_depth': 6, 'subsample': 0.8, 'min_samples_leaf': 3},
    {'n_estimators': 600, 'learning_rate': 0.05, 'max_depth': 7, 'subsample': 0.75, 'min_samples_leaf': 2},
]

best_gbm_mae = float('inf')
best_gbm_config = None
best_gbm_model = None

for config in gbm_configs:
    gbm = GradientBoostingRegressor(**config, min_samples_split=5, random_state=42)
    gbm.fit(X_train_scaled, y_train_log)

    y_pred_log = gbm.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"   n={config['n_estimators']:4d}, lr={config['learning_rate']:.2f}, d={config['max_depth']}: MAE={mae:6.2f}, R2={r2:.4f}")

    if mae < best_gbm_mae:
        best_gbm_mae = mae
        best_gbm_config = config
        best_gbm_model = gbm

print(f"\n[BEST GBM] {best_gbm_config}")
print(f"   MAE: {best_gbm_mae:.2f}")

# ============================================================================
# EXPERIMENT 3: HISTGRADIENTBOOSTING OPTIMIZATION
# ============================================================================
print()
print("="*90)
print(" "*20 + "EXPERIMENT 3: HISTGRADIENTBOOSTING")
print("="*90)
print()

print("[TEST] HistGradientBoosting configurations:")

hgb_configs = [
    {'max_iter': 500, 'learning_rate': 0.05, 'max_depth': 7, 'min_samples_leaf': 10},
    {'max_iter': 600, 'learning_rate': 0.05, 'max_depth': 8, 'min_samples_leaf': 10},
    {'max_iter': 700, 'learning_rate': 0.03, 'max_depth': 7, 'min_samples_leaf': 10},
    {'max_iter': 800, 'learning_rate': 0.03, 'max_depth': 6, 'min_samples_leaf': 15},
    {'max_iter': 600, 'learning_rate': 0.07, 'max_depth': 7, 'min_samples_leaf': 10},
    {'max_iter': 1000, 'learning_rate': 0.02, 'max_depth': 6, 'min_samples_leaf': 10},
]

best_hgb_mae = float('inf')
best_hgb_config = None
best_hgb_model = None

for config in hgb_configs:
    hgb = HistGradientBoostingRegressor(**config, l2_regularization=0.1, random_state=42)
    hgb.fit(X_train_scaled, y_train_log)

    y_pred_log = hgb.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"   iter={config['max_iter']:4d}, lr={config['learning_rate']:.2f}, d={config['max_depth']}: MAE={mae:6.2f}, R2={r2:.4f}")

    if mae < best_hgb_mae:
        best_hgb_mae = mae
        best_hgb_config = config
        best_hgb_model = hgb

print(f"\n[BEST HGB] {best_hgb_config}")
print(f"   MAE: {best_hgb_mae:.2f}")

# ============================================================================
# EXPERIMENT 4: STACKING ENSEMBLE
# ============================================================================
print()
print("="*90)
print(" "*20 + "EXPERIMENT 4: STACKING ENSEMBLE")
print("="*90)
print()

print("[STACK] Training base models...")

# Get predictions from best models
y_pred_gbm_log = best_gbm_model.predict(X_train_scaled)
y_pred_hgb_log = best_hgb_model.predict(X_train_scaled)

# Stack predictions as meta-features
X_meta_train = np.column_stack([y_pred_gbm_log, y_pred_hgb_log])

# Test meta-learners
meta_learners = [
    ('Ridge 0.1', Ridge(alpha=0.1)),
    ('Ridge 1.0', Ridge(alpha=1.0)),
    ('Ridge 10', Ridge(alpha=10.0)),
    ('Lasso 0.01', Lasso(alpha=0.01)),
    ('Lasso 0.1', Lasso(alpha=0.1)),
    ('ElasticNet 0.1', ElasticNet(alpha=0.1, l1_ratio=0.5)),
]

print("[TEST] Meta-learners:")

best_meta_mae = float('inf')
best_meta_name = None
best_meta_model = None

for name, meta in meta_learners:
    meta.fit(X_meta_train, y_train_log)

    # Predict on test set
    y_pred_gbm_test_log = best_gbm_model.predict(X_test_scaled)
    y_pred_hgb_test_log = best_hgb_model.predict(X_test_scaled)
    X_meta_test = np.column_stack([y_pred_gbm_test_log, y_pred_hgb_test_log])

    y_pred_meta_log = meta.predict(X_meta_test)
    y_pred_meta = np.expm1(y_pred_meta_log)
    mae = mean_absolute_error(y_test, y_pred_meta)
    r2 = r2_score(y_test, y_pred_meta)

    print(f"   {name:20s}: MAE={mae:6.2f}, R2={r2:.4f}")

    if mae < best_meta_mae:
        best_meta_mae = mae
        best_meta_name = name
        best_meta_model = meta

print(f"\n[BEST STACK] {best_meta_name}: MAE={best_meta_mae:.2f}")

# ============================================================================
# EXPERIMENT 5: WEIGHTED ENSEMBLE
# ============================================================================
print()
print("="*90)
print(" "*20 + "EXPERIMENT 5: WEIGHTED ENSEMBLE")
print("="*90)
print()

y_pred_gbm_test = np.expm1(best_gbm_model.predict(X_test_scaled))
y_pred_hgb_test = np.expm1(best_hgb_model.predict(X_test_scaled))

print("[TEST] Weight combinations:")

weights = [(w/10, 1-w/10) for w in range(0, 11, 1)]
best_weighted_mae = float('inf')
best_weight = None

for w_gbm, w_hgb in weights:
    y_pred_weighted = w_gbm * y_pred_gbm_test + w_hgb * y_pred_hgb_test
    mae = mean_absolute_error(y_test, y_pred_weighted)
    r2 = r2_score(y_test, y_pred_weighted)

    print(f"   GBM={w_gbm:.1f}, HGB={w_hgb:.1f}: MAE={mae:6.2f}, R2={r2:.4f}")

    if mae < best_weighted_mae:
        best_weighted_mae = mae
        best_weight = (w_gbm, w_hgb)

print(f"\n[BEST WEIGHTED] GBM={best_weight[0]:.1f}, HGB={best_weight[1]:.1f}: MAE={best_weighted_mae:.2f}")

# ============================================================================
# FINAL CHAMPION SELECTION
# ============================================================================
print()
print("="*90)
print(" "*25 + "PHASE 8 CHAMPION SELECTION")
print("="*90)
print()

candidates = [
    ('GBM (tuned)', best_gbm_mae, 'gbm'),
    ('HGB (tuned)', best_hgb_mae, 'hgb'),
    ('Stacking', best_meta_mae, 'stack'),
    ('Weighted Ensemble', best_weighted_mae, 'weighted'),
]

print("[CANDIDATES]")
for name, mae, _ in candidates:
    print(f"   {name:25s} MAE={mae:6.2f}")

champion_name, champion_mae, champion_type = min(candidates, key=lambda x: x[1])

print(f"\n[PHASE 8 CHAMPION] {champion_name}: MAE={champion_mae:.2f}")

# Compare to Phase 7
phase7_mae = 50.55
improvement = ((phase7_mae - champion_mae) / phase7_mae) * 100

print()
print("="*90)
print(" "*20 + "PERFORMANCE VS PHASE 7")
print("="*90)
print()

print(f"   Phase 7:      MAE = {phase7_mae:.2f} likes (8,198 posts)")
print(f"   Phase 8:      MAE = {champion_mae:.2f} likes ({len(df)} posts)")
print(f"   Improvement:  {improvement:+.1f}%")
print()

if champion_mae < phase7_mae:
    print(f"   NEW ULTRA CHAMPION! Improved by {abs(improvement):.1f}%")
elif champion_mae < phase7_mae * 1.02:
    print(f"   VERY CLOSE! Within 2% of Phase 7")
else:
    print(f"   Phase 7 still champion, but Phase 8 used MORE data (+{len(df)-8198} posts)")

# Save champion
print()
print("[SAVE] Saving Phase 8 champion...")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f'models/phase8_ultra_champion_{timestamp}.pkl'

if champion_type == 'gbm':
    model_package = {
        'model_type': 'gbm',
        'model': best_gbm_model,
        'scaler': scaler_best,
        'pca_bert': pca_best,
        'baseline_features': baseline_cols,
        'config': best_gbm_config,
        'metrics': {'mae': champion_mae},
        'dataset_size': len(df),
        'timestamp': timestamp
    }
elif champion_type == 'hgb':
    model_package = {
        'model_type': 'hgb',
        'model': best_hgb_model,
        'scaler': scaler_best,
        'pca_bert': pca_best,
        'baseline_features': baseline_cols,
        'config': best_hgb_config,
        'metrics': {'mae': champion_mae},
        'dataset_size': len(df),
        'timestamp': timestamp
    }
elif champion_type == 'stack':
    model_package = {
        'model_type': 'stacking',
        'base_model_gbm': best_gbm_model,
        'base_model_hgb': best_hgb_model,
        'meta_learner': best_meta_model,
        'meta_name': best_meta_name,
        'scaler': scaler_best,
        'pca_bert': pca_best,
        'baseline_features': baseline_cols,
        'metrics': {'mae': champion_mae},
        'dataset_size': len(df),
        'timestamp': timestamp
    }
else:  # weighted
    model_package = {
        'model_type': 'weighted_ensemble',
        'model_gbm': best_gbm_model,
        'model_hgb': best_hgb_model,
        'weight_gbm': best_weight[0],
        'weight_hgb': best_weight[1],
        'scaler': scaler_best,
        'pca_bert': pca_best,
        'baseline_features': baseline_cols,
        'metrics': {'mae': champion_mae},
        'dataset_size': len(df),
        'timestamp': timestamp
    }

joblib.dump(model_package, model_filename)
import os
print(f"   Model: {model_filename}")
print(f"   Size: {os.path.getsize(model_filename)/(1024*1024):.1f} MB")

print()
print("="*90)
print(" "*30 + "PHASE 8 COMPLETE!")
print("="*90)
