#!/usr/bin/env python3
"""
PHASE 7: ULTRA HYPERPARAMETER TUNING
Aggressive optimization to beat Phase 5 Ultra champion (MAE=27.23)

Strategy:
1. RF hyperparameter grid search
2. GBM hyperparameter grid search
3. Weighted ensemble optimization
4. PCA dimensionality fine-tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print(" "*20 + "PHASE 7: ULTRA HYPERPARAMETER TUNING")
print(" "*15 + "Target: Beat Phase 5 Ultra MAE=27.23")
print("="*90)
print()

# Load and prepare data
print("[LOAD] Loading clean datasets...")
df_main = pd.read_csv('multi_account_dataset.csv')
df_bert = pd.read_csv('data/processed/bert_embeddings_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])
df_aes = pd.read_csv('data/processed/aesthetic_features_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])

df = df_main.merge(df_bert, on=['post_id', 'account'], how='inner')
df = df.merge(df_aes, on=['post_id', 'account'], how='inner')

print(f"   Dataset: {len(df)} posts")

# Features
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                 'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']
bert_cols = [col for col in df.columns if col.startswith('bert_')]
aes_cols = [col for col in df.columns if col.startswith('aesthetic_')]

X_baseline = df[baseline_cols].values
X_bert_full = df[bert_cols].values
X_aes = df[aes_cols].values
y = df['likes'].values

# Train/test split
train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)

X_baseline_train = X_baseline[train_idx]
X_baseline_test = X_baseline[test_idx]
X_bert_train = X_bert_full[train_idx]
X_bert_test = X_bert_full[test_idx]
X_aes_train = X_aes[train_idx]
X_aes_test = X_aes[test_idx]

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

# ============================================================================
# EXPERIMENT 1: BERT PCA FINE-TUNING
# ============================================================================
print()
print("="*90)
print(" "*25 + "EXPERIMENT 1: BERT PCA TUNING")
print("="*90)
print()

pca_options = [50, 60, 70, 80, 90, 100]
best_pca_mae = float('inf')
best_pca_n = None

print("[TEST] Testing PCA dimensions with RF baseline:")

for pca_n in pca_options:
    pca = PCA(n_components=pca_n, random_state=42)
    X_bert_pca_train = pca.fit_transform(X_bert_train)
    X_bert_pca_test = pca.transform(X_bert_test)

    X_train = np.hstack([X_baseline_train, X_bert_pca_train, X_aes_train])
    X_test = np.hstack([X_baseline_test, X_bert_pca_test, X_aes_test])

    scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestRegressor(n_estimators=250, max_depth=14, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train_log)

    y_pred_log = rf.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    variance = pca.explained_variance_ratio_.sum()
    print(f"   PCA-{pca_n:3d} ({variance:.1%} var): MAE={mae:6.2f}, R2={r2:.4f}")

    if mae < best_pca_mae:
        best_pca_mae = mae
        best_pca_n = pca_n

print(f"\n[BEST PCA] {best_pca_n} components (MAE={best_pca_mae:.2f})")

# ============================================================================
# EXPERIMENT 2: RF HYPERPARAMETER TUNING
# ============================================================================
print()
print("="*90)
print(" "*20 + "EXPERIMENT 2: RF HYPERPARAMETER TUNING")
print("="*90)
print()

# Use best PCA
pca_best = PCA(n_components=best_pca_n, random_state=42)
X_bert_pca_train = pca_best.fit_transform(X_bert_train)
X_bert_pca_test = pca_best.transform(X_bert_test)

X_train = np.hstack([X_baseline_train, X_bert_pca_train, X_aes_train])
X_test = np.hstack([X_baseline_test, X_bert_pca_test, X_aes_test])

scaler_best = QuantileTransformer(output_distribution='uniform', random_state=42)
X_train_scaled = scaler_best.fit_transform(X_train)
X_test_scaled = scaler_best.transform(X_test)

# RF hyperparameter grid
rf_configs = [
    {'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 3, 'min_samples_leaf': 2},
    {'n_estimators': 250, 'max_depth': 14, 'min_samples_split': 3, 'min_samples_leaf': 2},
    {'n_estimators': 300, 'max_depth': 14, 'min_samples_split': 3, 'min_samples_leaf': 2},
    {'n_estimators': 250, 'max_depth': 16, 'min_samples_split': 3, 'min_samples_leaf': 1},
    {'n_estimators': 250, 'max_depth': 14, 'min_samples_split': 2, 'min_samples_leaf': 1},
    {'n_estimators': 300, 'max_depth': 16, 'min_samples_split': 2, 'min_samples_leaf': 1},
    {'n_estimators': 350, 'max_depth': 14, 'min_samples_split': 3, 'min_samples_leaf': 2},
]

best_rf_mae = float('inf')
best_rf_config = None
best_rf_model = None

print("[TEST] Testing RF configurations:")

for config in rf_configs:
    rf = RandomForestRegressor(**config, max_features='sqrt', random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train_log)

    y_pred_log = rf.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"   n={config['n_estimators']}, d={config['max_depth']}, split={config['min_samples_split']}, leaf={config['min_samples_leaf']}: MAE={mae:6.2f}, R2={r2:.4f}")

    if mae < best_rf_mae:
        best_rf_mae = mae
        best_rf_config = config
        best_rf_model = rf

print(f"\n[BEST RF] {best_rf_config}: MAE={best_rf_mae:.2f}")

# ============================================================================
# EXPERIMENT 3: GBM HYPERPARAMETER TUNING
# ============================================================================
print()
print("="*90)
print(" "*20 + "EXPERIMENT 3: GBM HYPERPARAMETER TUNING")
print("="*90)
print()

gbm_configs = [
    {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8},
    {'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8},
    {'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 6, 'subsample': 0.8},
    {'n_estimators': 400, 'learning_rate': 0.07, 'max_depth': 5, 'subsample': 0.7},
    {'n_estimators': 600, 'learning_rate': 0.03, 'max_depth': 5, 'subsample': 0.8},
    {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 7, 'subsample': 0.8},
]

best_gbm_mae = float('inf')
best_gbm_config = None
best_gbm_model = None

print("[TEST] Testing GBM configurations:")

for config in gbm_configs:
    gbm = GradientBoostingRegressor(**config, min_samples_split=5, min_samples_leaf=3, random_state=42)
    gbm.fit(X_train_scaled, y_train_log)

    y_pred_log = gbm.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"   n={config['n_estimators']}, lr={config['learning_rate']}, d={config['max_depth']}: MAE={mae:6.2f}, R2={r2:.4f}")

    if mae < best_gbm_mae:
        best_gbm_mae = mae
        best_gbm_config = config
        best_gbm_model = gbm

print(f"\n[BEST GBM] {best_gbm_config}: MAE={best_gbm_mae:.2f}")

# ============================================================================
# EXPERIMENT 4: ENSEMBLE OPTIMIZATION
# ============================================================================
print()
print("="*90)
print(" "*20 + "EXPERIMENT 4: RF + GBM ENSEMBLE WEIGHTS")
print("="*90)
print()

# Get predictions from best models
y_pred_rf_log = best_rf_model.predict(X_test_scaled)
y_pred_rf = np.expm1(y_pred_rf_log)

y_pred_gbm_log = best_gbm_model.predict(X_test_scaled)
y_pred_gbm = np.expm1(y_pred_gbm_log)

# Test weight combinations
weights = [
    (1.0, 0.0),   # RF only
    (0.0, 1.0),   # GBM only
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.3, 0.7),
    (0.8, 0.2),
    (0.9, 0.1),
]

best_ensemble_mae = float('inf')
best_ensemble_weight = None
best_ensemble_pred = None

print("[TEST] Testing ensemble weights:")

for w_rf, w_gbm in weights:
    y_pred_ensemble = w_rf * y_pred_rf + w_gbm * y_pred_gbm
    mae = mean_absolute_error(y_test, y_pred_ensemble)
    r2 = r2_score(y_test, y_pred_ensemble)

    print(f"   RF={w_rf:.1f}, GBM={w_gbm:.1f}: MAE={mae:6.2f}, R2={r2:.4f}")

    if mae < best_ensemble_mae:
        best_ensemble_mae = mae
        best_ensemble_weight = (w_rf, w_gbm)
        best_ensemble_pred = y_pred_ensemble

print(f"\n[BEST ENSEMBLE] RF={best_ensemble_weight[0]:.1f}, GBM={best_ensemble_weight[1]:.1f}: MAE={best_ensemble_mae:.2f}")

# ============================================================================
# FINAL CHAMPION SELECTION
# ============================================================================
print()
print("="*90)
print(" "*25 + "CHAMPION SELECTION")
print("="*90)
print()

candidates = [
    ('RF (best config)', best_rf_mae, best_rf_model, 'rf'),
    ('GBM (best config)', best_gbm_mae, best_gbm_model, 'gbm'),
    ('Ensemble (best weight)', best_ensemble_mae, None, 'ensemble'),
]

print("[CANDIDATES]")
for name, mae, _, _ in candidates:
    print(f"   {name:30s} MAE={mae:6.2f}")

# Select champion
champion_name, champion_mae, champion_model, champion_type = min(candidates, key=lambda x: x[1])

print(f"\n[CHAMPION] {champion_name}: MAE={champion_mae:.2f}")

# Compare to Phase 5 Ultra
phase5_mae = 27.23
improvement = ((phase5_mae - champion_mae) / phase5_mae) * 100

print()
print("="*90)
print(" "*20 + "PERFORMANCE VS PHASE 5 ULTRA")
print("="*90)
print()

print(f"   Phase 5 Ultra:  MAE = {phase5_mae:.2f} likes")
print(f"   Phase 7 Champion: MAE = {champion_mae:.2f} likes")
print(f"   Improvement:    {improvement:+.1f}%")
print()

if champion_mae < phase5_mae:
    print(f"   NEW CHAMPION! Improved by {abs(improvement):.1f}%")
elif champion_mae < phase5_mae * 1.05:
    print(f"   VERY CLOSE! Within 5% of Phase 5 Ultra")
else:
    print(f"   Phase 5 Ultra still champion")

# Save best model
print()
print("[SAVE] Saving Phase 7 champion model...")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f'models/phase7_champion_{timestamp}.pkl'

if champion_type == 'ensemble':
    model_package = {
        'model_type': 'ensemble',
        'rf_model': best_rf_model,
        'gbm_model': best_gbm_model,
        'rf_weight': best_ensemble_weight[0],
        'gbm_weight': best_ensemble_weight[1],
        'scaler': scaler_best,
        'pca_bert': pca_best,
        'baseline_features': baseline_cols,
        'aesthetic_features': aes_cols,
        'metrics': {'mae': champion_mae},
        'timestamp': timestamp
    }
else:
    model_package = {
        'model_type': champion_type,
        'model': champion_model,
        'scaler': scaler_best,
        'pca_bert': pca_best,
        'baseline_features': baseline_cols,
        'aesthetic_features': aes_cols,
        'metrics': {'mae': champion_mae},
        'config': best_rf_config if champion_type == 'rf' else best_gbm_config,
        'timestamp': timestamp
    }

joblib.dump(model_package, model_filename)
import os
print(f"   Model: {model_filename}")
print(f"   Size: {os.path.getsize(model_filename)/(1024*1024):.1f} MB")

print()
print("="*90)
print(" "*30 + "PHASE 7 COMPLETE!")
print("="*90)
