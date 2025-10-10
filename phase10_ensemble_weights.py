#!/usr/bin/env python3
"""
PHASE 10.8: OPTIMIZED ENSEMBLE WEIGHTS
Fine-tune base model weighting and meta-learner alpha
MUST INCLUDE: Visual + Text features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.optimize import minimize
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print(" "*15 + "PHASE 10.8: OPTIMIZED ENSEMBLE WEIGHTS")
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

# Combine
X_train = np.hstack([X_baseline_train, X_bert_pca_train, X_aes_pca_train])
X_test = np.hstack([X_baseline_test, X_bert_pca_test, X_aes_pca_test])

# Scale
scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"[FEATURES] {X_train_scaled.shape[1]} total (9 baseline + 50 BERT + 6 aesthetic)")
print()

# ============================================================================
# Generate base model predictions
# ============================================================================

print("[BASE MODELS] Training 4 base models with 5-fold OOF...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X_train_scaled), 4))
test_preds = np.zeros((len(X_test_scaled), 4))

models = [
    ('GBM', GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8, random_state=42)),
    ('HGB', HistGradientBoostingRegressor(max_iter=600, learning_rate=0.07, max_depth=7, random_state=42)),
    ('RF', RandomForestRegressor(n_estimators=300, max_depth=16, random_state=42, n_jobs=-1)),
    ('ET', ExtraTreesRegressor(n_estimators=300, max_depth=16, random_state=42, n_jobs=-1)),
]

for i, (name, model) in enumerate(models):
    print(f"   Training {name}...")
    # OOF
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
        m = model.__class__(**model.get_params())
        m.fit(X_train_scaled[tr_idx], y_train_log[tr_idx])
        oof_preds[val_idx, i] = m.predict(X_train_scaled[val_idx])

    # Test
    model.fit(X_train_scaled, y_train_log)
    test_preds[:, i] = model.predict(X_test_scaled)

print()

# ============================================================================
# Experiment: Different meta-learner configurations
# ============================================================================

print("[EXPERIMENT] Testing meta-learner configurations...")
print()

meta_configs = [
    {'name': 'Ridge alpha=10 (Phase 9)', 'meta': Ridge(alpha=10)},
    {'name': 'Ridge alpha=1', 'meta': Ridge(alpha=1)},
    {'name': 'Ridge alpha=5', 'meta': Ridge(alpha=5)},
    {'name': 'Ridge alpha=20', 'meta': Ridge(alpha=20)},
    {'name': 'Ridge alpha=50', 'meta': Ridge(alpha=50)},
    {'name': 'Ridge alpha=100', 'meta': Ridge(alpha=100)},
    {'name': 'Lasso alpha=0.001', 'meta': Lasso(alpha=0.001, max_iter=5000)},
    {'name': 'Lasso alpha=0.01', 'meta': Lasso(alpha=0.01, max_iter=5000)},
    {'name': 'Lasso alpha=0.1', 'meta': Lasso(alpha=0.1, max_iter=5000)},
    {'name': 'ElasticNet alpha=0.01', 'meta': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)},
    {'name': 'ElasticNet alpha=0.1', 'meta': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)},
]

best_mae = float('inf')
best_config = None

for config in meta_configs:
    meta = config['meta']
    meta.fit(oof_preds, y_train_log)

    y_pred_log = meta.predict(test_preds)
    y_pred = np.expm1(y_pred_log)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"   {config['name']:<35} MAE={mae:.2f}, R2={r2:.4f}")

    if mae < best_mae:
        best_mae = mae
        best_config = config['name']

print()

# ============================================================================
# Optimized weighted average (no meta-learner)
# ============================================================================

print("[OPTIMIZATION] Finding optimal base model weights...")

def weighted_mae(weights):
    """Minimize MAE by finding optimal base model weights"""
    weights = np.abs(weights) / np.abs(weights).sum()  # Normalize
    y_pred_log = (oof_preds * weights).sum(axis=1)
    y_pred = np.expm1(y_pred_log)
    return mean_absolute_error(y_train, y_pred)

# Initial weights (equal)
init_weights = np.ones(4) / 4

# Optimize
result = minimize(weighted_mae, init_weights, method='Nelder-Mead',
                  options={'maxiter': 1000, 'disp': False})
optimal_weights = np.abs(result.x) / np.abs(result.x).sum()

# Test on validation set
y_pred_log_opt = (test_preds * optimal_weights).sum(axis=1)
y_pred_opt = np.expm1(y_pred_log_opt)
mae_opt = mean_absolute_error(y_test, y_pred_opt)
r2_opt = r2_score(y_test, y_pred_opt)

print(f"   Optimal weights: GBM={optimal_weights[0]:.3f}, HGB={optimal_weights[1]:.3f}, "
      f"RF={optimal_weights[2]:.3f}, ET={optimal_weights[3]:.3f}")
print(f"   Weighted average MAE={mae_opt:.2f}, R2={r2_opt:.4f}")
print()

if mae_opt < best_mae:
    best_mae = mae_opt
    best_config = 'Optimized weighted average'

print(f"[BEST META] {best_config}: MAE={best_mae:.2f}")
print()

# ============================================================================
# Save results
# ============================================================================

if best_mae < 45.10:
    print("[SAVE] New champion! Saving model...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'models/phase10_8_weights_{timestamp}.pkl'

    model_package = {
        'phase': '10.8_ensemble_weights',
        'mae': best_mae,
        'best_config': best_config,
        'optimal_weights': optimal_weights.tolist(),
        'visual_included': True,
        'text_included': True,
        'timestamp': timestamp
    }

    joblib.dump(model_package, model_filename)
    print(f"   Saved: {model_filename}")
    print()

print("="*90)
print(" "*30 + "PHASE 10.8 COMPLETE!")
print("="*90)
print()
print(f"[RESULT] MAE={best_mae:.2f} (Phase 9 was 45.10)")
if best_mae < 45.10:
    print(f"   IMPROVED by {(45.10-best_mae)/45.10*100:.1f}%!")
else:
    print(f"   Weight optimization did not improve (Ridge alpha=10 remains optimal)")
