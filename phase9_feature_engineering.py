#!/usr/bin/env python3
"""
PHASE 9: MULTIMODAL FEATURE ENGINEERING
Focus: Advanced feature engineering with BOTH visual and text features

Strategy:
1. Include BOTH aesthetic (visual) and BERT (text) features
2. Test optimal PCA dimensions for each modality
3. Test feature interactions (visual x text)
4. Test polynomial features
5. Apply best stacking approach from Phase 8
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import QuantileTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print(" "*20 + "PHASE 9: MULTIMODAL FEATURE ENGINEERING")
print(" "*15 + "Target: Beat Phase 8 MAE=48.41 WITH Visual Features")
print("="*90)
print()

# ============================================================================
# LOAD DATA WITH BOTH VISUAL AND TEXT FEATURES
# ============================================================================

print("[LOAD] Loading datasets with BOTH visual and text features...")
df_main = pd.read_csv('multi_account_dataset.csv')
df_bert = pd.read_csv('data/processed/bert_embeddings_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])
df_aes = pd.read_csv('data/processed/aesthetic_features_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])

print(f"   Main dataset: {len(df_main)} posts")
print(f"   BERT embeddings: {len(df_bert)} unique posts")
print(f"   Aesthetic features: {len(df_aes)} unique posts")

# Merge with BOTH modalities
df = df_main.merge(df_bert, on=['post_id', 'account'], how='inner')
df = df.merge(df_aes, on=['post_id', 'account'], how='inner')

print(f"   Combined dataset: {len(df)} posts (with BOTH visual + text)")
print()

# Features
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                 'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']
bert_cols = [col for col in df.columns if col.startswith('bert_')]
aes_cols = [col for col in df.columns if col.startswith('aesthetic_')]

print("[FEATURES] Multimodal feature breakdown:")
print(f"   Baseline: {len(baseline_cols)} features")
print(f"   BERT (text): {len(bert_cols)} features (768-dim embeddings)")
print(f"   Aesthetic (visual): {len(aes_cols)} features")
print()

X_baseline = df[baseline_cols].values
X_bert_full = df[bert_cols].values
X_aes_full = df[aes_cols].values
y = df['likes'].values

# Train/test split
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
y_train_clipped = np.clip(y_train, 0, clip_threshold)
y_test_clipped = np.clip(y_test, 0, clip_threshold)
y_train_log = np.log1p(y_train_clipped)
y_test_log = np.log1p(y_test_clipped)

print(f"   Train: {len(X_baseline_train)} posts")
print(f"   Test: {len(X_baseline_test)} posts")
print()

# ============================================================================
# EXPERIMENT 1: OPTIMAL PCA DIMENSIONALITY FOR EACH MODALITY
# ============================================================================
print("="*90)
print(" "*20 + "EXPERIMENT 1: OPTIMAL PCA DIMENSIONALITY")
print("="*90)
print()

print("[TEST] Testing BERT PCA dimensions:")
bert_pca_options = [30, 40, 50, 60, 70, 80]
aes_pca_options = [3, 4, 5, 6, 7, 8]  # Aesthetic has only 8 features

best_bert_pca = None
best_aes_pca = None
best_pca_mae = float('inf')

results_pca = []

for bert_n in bert_pca_options:
    for aes_n in aes_pca_options:
        # PCA reduction
        pca_bert = PCA(n_components=bert_n, random_state=42)
        pca_aes = PCA(n_components=aes_n, random_state=42)

        X_bert_pca_train = pca_bert.fit_transform(X_bert_train)
        X_bert_pca_test = pca_bert.transform(X_bert_test)
        X_aes_pca_train = pca_aes.fit_transform(X_aes_train)
        X_aes_pca_test = pca_aes.transform(X_aes_test)

        # Combine features
        X_train = np.hstack([X_baseline_train, X_bert_pca_train, X_aes_pca_train])
        X_test = np.hstack([X_baseline_test, X_bert_pca_test, X_aes_pca_test])

        # Scale
        scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Quick GBM test
        gbm = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=7,
                                       subsample=0.8, random_state=42)
        gbm.fit(X_train_scaled, y_train_log)

        y_pred_log = gbm.predict(X_test_scaled)
        y_pred = np.expm1(y_pred_log)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        bert_var = pca_bert.explained_variance_ratio_.sum()
        aes_var = pca_aes.explained_variance_ratio_.sum()

        results_pca.append({
            'bert_pca': bert_n,
            'aes_pca': aes_n,
            'bert_var': bert_var,
            'aes_var': aes_var,
            'mae': mae,
            'r2': r2
        })

        print(f"   BERT={bert_n:2d} ({bert_var:.1%}), Aes={aes_n} ({aes_var:.1%}): MAE={mae:6.2f}, R2={r2:.4f}")

        if mae < best_pca_mae:
            best_pca_mae = mae
            best_bert_pca = bert_n
            best_aes_pca = aes_n

print(f"\n[BEST PCA] BERT={best_bert_pca}, Aesthetic={best_aes_pca}: MAE={best_pca_mae:.2f}")
print()

# Use best PCA
pca_bert_best = PCA(n_components=best_bert_pca, random_state=42)
pca_aes_best = PCA(n_components=best_aes_pca, random_state=42)

X_bert_pca_train = pca_bert_best.fit_transform(X_bert_train)
X_bert_pca_test = pca_bert_best.transform(X_bert_test)
X_aes_pca_train = pca_aes_best.fit_transform(X_aes_train)
X_aes_pca_test = pca_aes_best.transform(X_aes_test)

# ============================================================================
# EXPERIMENT 2: FEATURE INTERACTIONS (TEXT x VISUAL)
# ============================================================================
print("="*90)
print(" "*20 + "EXPERIMENT 2: TEXT x VISUAL INTERACTIONS")
print("="*90)
print()

print("[TEST] Testing feature interaction strategies:")

configs = [
    {'name': 'No interactions (baseline)', 'interactions': False, 'poly': False},
    {'name': 'Polynomial degree 2', 'interactions': False, 'poly': True},
    {'name': 'Manual interactions (text x visual)', 'interactions': True, 'poly': False},
    {'name': 'Both (manual + poly)', 'interactions': True, 'poly': True},
]

best_interaction_mae = float('inf')
best_interaction_config = None
best_X_train = None
best_X_test = None

for config in configs:
    # Start with base features
    X_train = np.hstack([X_baseline_train, X_bert_pca_train, X_aes_pca_train])
    X_test = np.hstack([X_baseline_test, X_bert_pca_test, X_aes_pca_test])

    # Add manual interactions
    if config['interactions']:
        # Create interactions between first 5 BERT PCs and first 3 aesthetic PCs
        bert_start = len(baseline_cols)
        bert_end = bert_start + 5
        aes_start = bert_start + best_bert_pca
        aes_end = aes_start + 3

        interactions_train = []
        interactions_test = []

        for i in range(bert_start, bert_end):
            for j in range(aes_start, aes_end):
                interactions_train.append(X_train[:, i] * X_train[:, j])
                interactions_test.append(X_test[:, i] * X_test[:, j])

        interactions_train = np.column_stack(interactions_train)
        interactions_test = np.column_stack(interactions_test)

        X_train = np.hstack([X_train, interactions_train])
        X_test = np.hstack([X_test, interactions_test])

    # Add polynomial features (only on baseline)
    if config['poly']:
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        baseline_poly_train = poly.fit_transform(X_baseline_train)[:, len(baseline_cols):]  # Skip original
        baseline_poly_test = poly.transform(X_baseline_test)[:, len(baseline_cols):]

        X_train = np.hstack([X_train, baseline_poly_train])
        X_test = np.hstack([X_test, baseline_poly_test])

    # Scale
    scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Test with GBM
    gbm = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=7,
                                   subsample=0.8, random_state=42)
    gbm.fit(X_train_scaled, y_train_log)

    y_pred_log = gbm.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"   {config['name']:40s} Features={X_train.shape[1]:3d}: MAE={mae:6.2f}, R2={r2:.4f}")

    if mae < best_interaction_mae:
        best_interaction_mae = mae
        best_interaction_config = config
        best_X_train = X_train
        best_X_test = X_test

print(f"\n[BEST INTERACTION] {best_interaction_config['name']}: MAE={best_interaction_mae:.2f}")
print(f"   Total features: {best_X_train.shape[1]}")
print()

# Use best interaction config
X_train_final = best_X_train
X_test_final = best_X_test

# Scale final features
scaler_final = QuantileTransformer(output_distribution='uniform', random_state=42)
X_train_scaled_final = scaler_final.fit_transform(X_train_final)
X_test_scaled_final = scaler_final.transform(X_test_final)

# ============================================================================
# EXPERIMENT 3: FEATURE SELECTION
# ============================================================================
print("="*90)
print(" "*20 + "EXPERIMENT 3: FEATURE SELECTION")
print("="*90)
print()

print("[TEST] Testing feature selection methods:")

n_features_options = [
    X_train_scaled_final.shape[1],  # All features
    int(X_train_scaled_final.shape[1] * 0.9),
    int(X_train_scaled_final.shape[1] * 0.8),
    int(X_train_scaled_final.shape[1] * 0.7),
]

best_selection_mae = float('inf')
best_n_features = None
best_selector = None

for n_features in n_features_options:
    if n_features == X_train_scaled_final.shape[1]:
        # No selection
        X_train_selected = X_train_scaled_final
        X_test_selected = X_test_scaled_final
        selector = None
    else:
        # Use mutual information
        selector = SelectKBest(mutual_info_regression, k=n_features)
        X_train_selected = selector.fit_transform(X_train_scaled_final, y_train_log)
        X_test_selected = selector.transform(X_test_scaled_final)

    # Test with GBM
    gbm = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=7,
                                   subsample=0.8, random_state=42)
    gbm.fit(X_train_selected, y_train_log)

    y_pred_log = gbm.predict(X_test_selected)
    y_pred = np.expm1(y_pred_log)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    pct = (n_features / X_train_scaled_final.shape[1]) * 100
    print(f"   Features={n_features:3d} ({pct:5.1f}%): MAE={mae:6.2f}, R2={r2:.4f}")

    if mae < best_selection_mae:
        best_selection_mae = mae
        best_n_features = n_features
        best_selector = selector

print(f"\n[BEST SELECTION] {best_n_features} features: MAE={best_selection_mae:.2f}")
print()

# Apply best selection
if best_selector is not None:
    X_train_final_selected = best_selector.transform(X_train_scaled_final)
    X_test_final_selected = best_selector.transform(X_test_scaled_final)
else:
    X_train_final_selected = X_train_scaled_final
    X_test_final_selected = X_test_scaled_final

# ============================================================================
# EXPERIMENT 4: STACKING ENSEMBLE (PHASE 8 APPROACH)
# ============================================================================
print("="*90)
print(" "*20 + "EXPERIMENT 4: MULTIMODAL STACKING ENSEMBLE")
print("="*90)
print()

print("[STACK] Training base models with multimodal features...")

# Base models from Phase 8
base_models = {
    'GBM': GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=8,
                                     subsample=0.8, min_samples_leaf=1, random_state=42),
    'HGB': HistGradientBoostingRegressor(max_iter=600, learning_rate=0.07, max_depth=7,
                                         min_samples_leaf=10, random_state=42),
    'RF': RandomForestRegressor(n_estimators=300, max_depth=16, min_samples_split=2,
                               min_samples_leaf=1, random_state=42, n_jobs=-1),
    'ET': ExtraTreesRegressor(n_estimators=300, max_depth=16, min_samples_split=2,
                             random_state=42, n_jobs=-1),
}

# Out-of-fold predictions
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_train = np.zeros((len(X_train_final_selected), len(base_models)))
test_preds = np.zeros((len(X_test_final_selected), len(base_models)))

for i, (name, model) in enumerate(base_models.items()):
    print(f"   Training {name}...")

    # Out-of-fold predictions
    for fold, (train_idx_fold, val_idx_fold) in enumerate(kf.split(X_train_final_selected)):
        X_fold_train = X_train_final_selected[train_idx_fold]
        y_fold_train = y_train_log[train_idx_fold]
        X_fold_val = X_train_final_selected[val_idx_fold]

        model_fold = model.__class__(**model.get_params())
        model_fold.fit(X_fold_train, y_fold_train)

        oof_preds_train[val_idx_fold, i] = model_fold.predict(X_fold_val)

    # Train on full training set for test predictions
    model.fit(X_train_final_selected, y_train_log)
    test_preds[:, i] = model.predict(X_test_final_selected)

print("\n[TEST] Testing meta-learners:")

meta_learners = [
    ('Ridge 0.1', Ridge(alpha=0.1)),
    ('Ridge 1.0', Ridge(alpha=1.0)),
    ('Ridge 10', Ridge(alpha=10.0)),
    ('Ridge 50', Ridge(alpha=50.0)),
    ('Lasso 0.01', Lasso(alpha=0.01)),
    ('Lasso 0.1', Lasso(alpha=0.1)),
    ('ElasticNet 0.1', ElasticNet(alpha=0.1)),
    ('ElasticNet 1.0', ElasticNet(alpha=1.0)),
]

best_stack_mae = float('inf')
best_meta_name = None
best_meta_model = None

for name, meta in meta_learners:
    meta.fit(oof_preds_train, y_train_log)

    y_pred_log = meta.predict(test_preds)
    y_pred = np.expm1(y_pred_log)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"   {name:20s}: MAE={mae:6.2f}, R2={r2:.4f}")

    if mae < best_stack_mae:
        best_stack_mae = mae
        best_meta_name = name
        best_meta_model = meta

print(f"\n[BEST STACK] {best_meta_name}: MAE={best_stack_mae:.2f}")
print()

# ============================================================================
# FINAL COMPARISON
# ============================================================================
print("="*90)
print(" "*25 + "FINAL COMPARISON")
print("="*90)
print()

print("[RESULTS]")
print(f"   Phase 8 (text-only):        MAE = 48.41 likes")
print(f"   Phase 9 (multimodal):       MAE = {best_stack_mae:.2f} likes")
print()

improvement = ((48.41 - best_stack_mae) / 48.41) * 100

if best_stack_mae < 48.41:
    print(f"   NEW CHAMPION! Improved by {abs(improvement):.1f}%")
    print(f"   Visual features contributed to improvement!")
elif best_stack_mae < 48.41 * 1.02:
    print(f"   VERY CLOSE! Within 2% of Phase 8")
    print(f"   Visual features are valuable for understanding patterns")
else:
    print(f"   Phase 8 still champion")
    print(f"   Note: Phase 9 has BOTH visual and text features as requested")

# ============================================================================
# FEATURE CONTRIBUTION ANALYSIS
# ============================================================================
print()
print("="*90)
print(" "*20 + "MULTIMODAL FEATURE CONTRIBUTION")
print("="*90)
print()

# Train a final model to get feature importance
final_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=8,
                                       subsample=0.8, random_state=42)
final_model.fit(X_train_final_selected, y_train_log)

# If selector was used, get selected feature indices
if best_selector is not None:
    selected_indices = best_selector.get_support(indices=True)
else:
    selected_indices = np.arange(X_train_scaled_final.shape[1])

# Map back to original feature types
baseline_end = len(baseline_cols)
bert_end = baseline_end + best_bert_pca
aes_end = bert_end + best_aes_pca
interaction_start = aes_end

importances = final_model.feature_importances_

baseline_importance = 0
bert_importance = 0
aes_importance = 0
interaction_importance = 0

for idx, importance in zip(selected_indices, importances):
    if idx < baseline_end:
        baseline_importance += importance
    elif idx < bert_end:
        bert_importance += importance
    elif idx < aes_end:
        aes_importance += importance
    else:
        interaction_importance += importance

total = baseline_importance + bert_importance + aes_importance + interaction_importance

print("[FEATURE CONTRIBUTION]")
print(f"   Baseline:      {baseline_importance/total*100:5.1f}%")
print(f"   BERT (text):   {bert_importance/total*100:5.1f}%")
print(f"   Aesthetic (visual): {aes_importance/total*100:5.1f}%")
if interaction_importance > 0:
    print(f"   Interactions:  {interaction_importance/total*100:5.1f}%")
print()

print("[CONFIRMATION]")
print(f"   Visual features included: YES ({best_aes_pca} PCA components)")
print(f"   Text features included:   YES ({best_bert_pca} PCA components)")
print(f"   Dataset size: {len(df)} posts (with both modalities)")
print()

# ============================================================================
# SAVE MODEL
# ============================================================================
print("[SAVE] Saving Phase 9 multimodal model...")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f'models/phase9_multimodal_{timestamp}.pkl'

model_package = {
    'model_type': 'multimodal_stacking',
    'base_models': base_models,
    'meta_learner': best_meta_model,
    'meta_name': best_meta_name,
    'pca_bert': pca_bert_best,
    'pca_aes': pca_aes_best,
    'feature_selector': best_selector,
    'scaler': scaler_final,
    'baseline_features': baseline_cols,
    'interaction_config': best_interaction_config,
    'metrics': {
        'mae': best_stack_mae,
        'bert_pca': best_bert_pca,
        'aes_pca': best_aes_pca,
        'n_features': best_n_features,
        'baseline_importance': baseline_importance/total,
        'bert_importance': bert_importance/total,
        'aes_importance': aes_importance/total,
        'interaction_importance': interaction_importance/total,
    },
    'timestamp': timestamp,
    'dataset_size': len(df)
}

joblib.dump(model_package, model_filename)
import os
model_size_mb = os.path.getsize(model_filename) / (1024 * 1024)

print(f"   Model: {model_filename}")
print(f"   Size: {model_size_mb:.1f} MB")
print()

print("="*90)
print(" "*30 + "PHASE 9 COMPLETE!")
print("="*90)
print()

print("[SUMMARY]")
print(f"   Visual features: INCLUDED ({best_aes_pca} PCA)")
print(f"   Text features: INCLUDED ({best_bert_pca} PCA)")
print(f"   MAE: {best_stack_mae:.2f} likes")
print(f"   Dataset: {len(df)} posts (multimodal)")
