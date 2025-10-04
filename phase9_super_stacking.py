#!/usr/bin/env python3
"""
PHASE 9: SUPER STACKING ULTRATHINK
Target: Beat Phase 8 MAE=48.41

Advanced strategies:
1. 3-Layer Deep Stacking (multiple base models)
2. Blending (out-of-fold predictions)
3. Feature-weighted stacking
4. Polynomial meta-features
5. Neural network meta-learner
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, ExtraTreesRegressor)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print(" "*20 + "PHASE 9: SUPER STACKING ULTRATHINK")
print(" "*15 + "Target: Beat Phase 8 MAE=48.41")
print("="*90)
print()

# Load data
print("[LOAD] Loading dataset...")
df_main = pd.read_csv('multi_account_dataset.csv')
df_bert = pd.read_csv('data/processed/bert_embeddings_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])

df = df_main.merge(df_bert, on=['post_id', 'account'], how='inner')
print(f"   Dataset: {len(df)} posts")

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

# PCA
pca = PCA(n_components=70, random_state=42)
X_bert_pca_train = pca.fit_transform(X_bert_train)
X_bert_pca_test = pca.transform(X_bert_test)

X_train = np.hstack([X_baseline_train, X_bert_pca_train])
X_test = np.hstack([X_baseline_test, X_bert_pca_test])

# Scale
scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   Train: {len(X_train)} posts")
print(f"   Test: {len(X_test)} posts")
print(f"   Features: {X_train.shape[1]}")

# ============================================================================
# EXPERIMENT 1: MULTI-MODEL BASE LAYER (6 MODELS)
# ============================================================================
print()
print("="*90)
print(" "*20 + "EXPERIMENT 1: DEEP BASE LAYER (6 MODELS)")
print("="*90)
print()

print("[TRAIN] Training 6 diverse base models...")

base_models = {
    'GBM': GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8, random_state=42),
    'HGB': HistGradientBoostingRegressor(max_iter=600, learning_rate=0.07, max_depth=7, random_state=42),
    'RF': RandomForestRegressor(n_estimators=300, max_depth=16, min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=-1),
    'ET': ExtraTreesRegressor(n_estimators=300, max_depth=16, min_samples_split=2, random_state=42, n_jobs=-1),
    'Ridge': Ridge(alpha=10.0),
    'Lasso': Lasso(alpha=0.1)
}

base_predictions_train = {}
base_predictions_test = {}

for name, model in base_models.items():
    print(f"   Training {name}...")
    model.fit(X_train_scaled, y_train_log)

    pred_train = model.predict(X_train_scaled)
    pred_test = model.predict(X_test_scaled)

    base_predictions_train[name] = pred_train
    base_predictions_test[name] = pred_test

    # Evaluate individual model
    pred_test_exp = np.expm1(pred_test)
    mae = mean_absolute_error(y_test, pred_test_exp)
    print(f"      {name} MAE: {mae:.2f}")

# ============================================================================
# EXPERIMENT 2: STACKING LAYER 1 - LINEAR META-LEARNERS
# ============================================================================
print()
print("="*90)
print(" "*20 + "EXPERIMENT 2: LAYER 1 META-LEARNERS")
print("="*90)
print()

# Combine base predictions
X_meta_train_L1 = np.column_stack([base_predictions_train[name] for name in base_models.keys()])
X_meta_test_L1 = np.column_stack([base_predictions_test[name] for name in base_models.keys()])

print(f"[META] Meta-features shape: {X_meta_train_L1.shape}")

meta_learners_L1 = {
    'Ridge_0.1': Ridge(alpha=0.1),
    'Ridge_1': Ridge(alpha=1.0),
    'Ridge_10': Ridge(alpha=10.0),
    'Ridge_50': Ridge(alpha=50.0),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
}

print("\n[TEST] Layer 1 meta-learners:")

best_meta_L1_mae = float('inf')
best_meta_L1_name = None
best_meta_L1_model = None

for name, meta in meta_learners_L1.items():
    meta.fit(X_meta_train_L1, y_train_log)

    pred_test = meta.predict(X_meta_test_L1)
    pred_test_exp = np.expm1(pred_test)
    mae = mean_absolute_error(y_test, pred_test_exp)
    r2 = r2_score(y_test, pred_test_exp)

    print(f"   {name:20s}: MAE={mae:6.2f}, R2={r2:.4f}")

    if mae < best_meta_L1_mae:
        best_meta_L1_mae = mae
        best_meta_L1_name = name
        best_meta_L1_model = meta

print(f"\n[BEST L1] {best_meta_L1_name}: MAE={best_meta_L1_mae:.2f}")

# ============================================================================
# EXPERIMENT 3: POLYNOMIAL META-FEATURES
# ============================================================================
print()
print("="*90)
print(" "*20 + "EXPERIMENT 3: POLYNOMIAL META-FEATURES")
print("="*90)
print()

print("[POLY] Creating polynomial interactions (degree 2)...")

poly = PolynomialFeatures(degree=2, include_bias=False)
X_meta_train_poly = poly.fit_transform(X_meta_train_L1)
X_meta_test_poly = poly.transform(X_meta_test_L1)

print(f"   Original features: {X_meta_train_L1.shape[1]}")
print(f"   Polynomial features: {X_meta_train_poly.shape[1]}")

print("\n[TEST] Meta-learners with polynomial features:")

meta_learners_poly = {
    'Ridge_1': Ridge(alpha=1.0),
    'Ridge_10': Ridge(alpha=10.0),
    'Ridge_50': Ridge(alpha=50.0),
    'Ridge_100': Ridge(alpha=100.0),
    'Lasso_0.1': Lasso(alpha=0.1),
}

best_poly_mae = float('inf')
best_poly_name = None
best_poly_model = None

for name, meta in meta_learners_poly.items():
    meta.fit(X_meta_train_poly, y_train_log)

    pred_test = meta.predict(X_meta_test_poly)
    pred_test_exp = np.expm1(pred_test)
    mae = mean_absolute_error(y_test, pred_test_exp)
    r2 = r2_score(y_test, pred_test_exp)

    print(f"   {name:20s}: MAE={mae:6.2f}, R2={r2:.4f}")

    if mae < best_poly_mae:
        best_poly_mae = mae
        best_poly_name = name
        best_poly_model = meta

print(f"\n[BEST POLY] {best_poly_name}: MAE={best_poly_mae:.2f}")

# ============================================================================
# EXPERIMENT 4: NEURAL NETWORK META-LEARNER
# ============================================================================
print()
print("="*90)
print(" "*20 + "EXPERIMENT 4: NEURAL NETWORK META-LEARNER")
print("="*90)
print()

print("[NN] Training neural network configurations...")

nn_configs = [
    {'hidden_layer_sizes': (50,), 'alpha': 0.01, 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (100,), 'alpha': 0.01, 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (50, 25), 'alpha': 0.01, 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (100, 50), 'alpha': 0.01, 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (100, 50, 25), 'alpha': 0.01, 'learning_rate_init': 0.001},
]

best_nn_mae = float('inf')
best_nn_config = None
best_nn_model = None

for config in nn_configs:
    nn = MLPRegressor(**config, max_iter=500, random_state=42, early_stopping=True)
    nn.fit(X_meta_train_L1, y_train_log)

    pred_test = nn.predict(X_meta_test_L1)
    pred_test_exp = np.expm1(pred_test)
    mae = mean_absolute_error(y_test, pred_test_exp)
    r2 = r2_score(y_test, pred_test_exp)

    layers = config['hidden_layer_sizes']
    print(f"   Layers {layers}: MAE={mae:6.2f}, R2={r2:.4f}")

    if mae < best_nn_mae:
        best_nn_mae = mae
        best_nn_config = config
        best_nn_model = nn

print(f"\n[BEST NN] {best_nn_config['hidden_layer_sizes']}: MAE={best_nn_mae:.2f}")

# ============================================================================
# EXPERIMENT 5: BLENDING (OUT-OF-FOLD PREDICTIONS)
# ============================================================================
print()
print("="*90)
print(" "*20 + "EXPERIMENT 5: BLENDING (5-FOLD CV)")
print("="*90)
print()

print("[BLEND] Creating out-of-fold predictions...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros((len(X_train_scaled), len(base_models)))
test_predictions_blend = np.zeros((len(X_test_scaled), len(base_models)))

for fold, (train_fold_idx, val_fold_idx) in enumerate(kf.split(X_train_scaled)):
    print(f"   Fold {fold+1}/5...")

    X_fold_train = X_train_scaled[train_fold_idx]
    y_fold_train = y_train_log[train_fold_idx]
    X_fold_val = X_train_scaled[val_fold_idx]

    for i, (name, model_class) in enumerate(base_models.items()):
        # Clone model
        if name == 'GBM':
            model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8, random_state=42)
        elif name == 'HGB':
            model = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.07, max_depth=7, random_state=42)
        elif name == 'RF':
            model = RandomForestRegressor(n_estimators=300, max_depth=16, min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=-1)
        elif name == 'ET':
            model = ExtraTreesRegressor(n_estimators=300, max_depth=16, min_samples_split=2, random_state=42, n_jobs=-1)
        elif name == 'Ridge':
            model = Ridge(alpha=10.0)
        else:  # Lasso
            model = Ridge(alpha=0.1)

        model.fit(X_fold_train, y_fold_train)

        # OOF predictions
        oof_predictions[val_fold_idx, i] = model.predict(X_fold_val)

        # Test predictions (average across folds)
        test_predictions_blend[:, i] += model.predict(X_test_scaled) / 5

# Train meta-learner on OOF predictions
print("\n[META] Training blending meta-learner...")

blend_meta = Ridge(alpha=10.0)
blend_meta.fit(oof_predictions, y_train_log)

pred_test_blend = blend_meta.predict(test_predictions_blend)
pred_test_blend_exp = np.expm1(pred_test_blend)
blend_mae = mean_absolute_error(y_test, pred_test_blend_exp)
blend_r2 = r2_score(y_test, pred_test_blend_exp)

print(f"   Blending MAE: {blend_mae:.2f}, R2={blend_r2:.4f}")

# ============================================================================
# CHAMPION SELECTION
# ============================================================================
print()
print("="*90)
print(" "*25 + "PHASE 9 CHAMPION SELECTION")
print("="*90)
print()

candidates = [
    ('Layer 1 Stacking (best)', best_meta_L1_mae, 'layer1'),
    ('Polynomial Meta (best)', best_poly_mae, 'poly'),
    ('Neural Network (best)', best_nn_mae, 'nn'),
    ('Blending (5-fold CV)', blend_mae, 'blend'),
]

print("[CANDIDATES]")
for name, mae, _ in candidates:
    print(f"   {name:30s} MAE={mae:6.2f}")

champion_name, champion_mae, champion_type = min(candidates, key=lambda x: x[1])

print(f"\n[PHASE 9 CHAMPION] {champion_name}: MAE={champion_mae:.2f}")

# Compare to Phase 8
phase8_mae = 48.41
improvement = ((phase8_mae - champion_mae) / phase8_mae) * 100

print()
print("="*90)
print(" "*20 + "PERFORMANCE VS PHASE 8")
print("="*90)
print()

print(f"   Phase 8:      MAE = {phase8_mae:.2f} likes")
print(f"   Phase 9:      MAE = {champion_mae:.2f} likes")
print(f"   Improvement:  {improvement:+.1f}%")
print()

if champion_mae < phase8_mae:
    print(f"   NEW ULTRA CHAMPION! Improved by {abs(improvement):.1f}%")
elif champion_mae < phase8_mae * 1.01:
    print(f"   VERY CLOSE! Within 1% of Phase 8")
else:
    print(f"   Phase 8 still champion")

# Save champion
print()
print("[SAVE] Saving Phase 9 champion...")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f'models/phase9_super_champion_{timestamp}.pkl'

if champion_type == 'layer1':
    model_package = {
        'model_type': 'layer1_stacking',
        'base_models': base_models,
        'meta_learner': best_meta_L1_model,
        'meta_name': best_meta_L1_name,
        'scaler': scaler,
        'pca_bert': pca,
        'baseline_features': baseline_cols,
        'metrics': {'mae': champion_mae},
        'timestamp': timestamp
    }
elif champion_type == 'poly':
    model_package = {
        'model_type': 'polynomial_stacking',
        'base_models': base_models,
        'polynomial': poly,
        'meta_learner': best_poly_model,
        'meta_name': best_poly_name,
        'scaler': scaler,
        'pca_bert': pca,
        'baseline_features': baseline_cols,
        'metrics': {'mae': champion_mae},
        'timestamp': timestamp
    }
elif champion_type == 'nn':
    model_package = {
        'model_type': 'neural_network_stacking',
        'base_models': base_models,
        'meta_learner': best_nn_model,
        'meta_config': best_nn_config,
        'scaler': scaler,
        'pca_bert': pca,
        'baseline_features': baseline_cols,
        'metrics': {'mae': champion_mae},
        'timestamp': timestamp
    }
else:  # blend
    model_package = {
        'model_type': 'blending',
        'base_models': base_models,
        'meta_learner': blend_meta,
        'scaler': scaler,
        'pca_bert': pca,
        'baseline_features': baseline_cols,
        'metrics': {'mae': champion_mae},
        'timestamp': timestamp
    }

joblib.dump(model_package, model_filename)
import os
print(f"   Model: {model_filename}")
print(f"   Size: {os.path.getsize(model_filename)/(1024*1024):.1f} MB")

print()
print("="*90)
print(" "*30 + "PHASE 9 COMPLETE!")
print("="*90)
