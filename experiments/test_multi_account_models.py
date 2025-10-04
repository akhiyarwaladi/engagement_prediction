#!/usr/bin/env python3
"""
Multi-Account Model Training & Comparison
Test various feature combinations with 1,949 posts (vs 348 previous)

Experiments:
1. Baseline only (9 features)
2. Baseline + BERT PCA50 (59 features)
3. Baseline + BERT + NIMA (67 features)
4. Baseline + BERT + NIMA + RFE optimized

Expected: MAE < 120 (vs 135.21 with 348 posts)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print(" " * 20 + "MULTI-ACCOUNT MODEL TRAINING")
print(" " * 15 + "1,949 Posts vs 348 Posts Comparison")
print("=" * 80 + "\n")

# Load data
print("[LOAD] Loading datasets...")
baseline_df = pd.read_csv('multi_account_dataset.csv')
bert_df = pd.read_csv('data/processed/bert_embeddings_multi_account.csv')
nima_df = pd.read_csv('data/processed/aesthetic_features_multi_account.csv')

print(f"   Baseline: {len(baseline_df)} posts")
print(f"   BERT: {len(bert_df)} embeddings")
print(f"   NIMA: {len(nima_df)} aesthetic features")

# Merge all features
print("\n[MERGE] Merging features...")
merged_df = baseline_df.copy()

# Merge BERT embeddings
bert_features = bert_df[[c for c in bert_df.columns if c.startswith('bert_dim_')]]
merged_df = pd.concat([merged_df.reset_index(drop=True), bert_features.reset_index(drop=True)], axis=1)

# Merge NIMA features
nima_features = nima_df[[c for c in nima_df.columns if c.startswith('aesthetic_')]]
merged_df = pd.concat([merged_df.reset_index(drop=True), nima_features.reset_index(drop=True)], axis=1)

print(f"   Merged shape: {merged_df.shape}")

# Prepare features
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                 'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']

bert_cols = [c for c in merged_df.columns if c.startswith('bert_dim_')]
nima_cols = [c for c in merged_df.columns if c.startswith('aesthetic_')]

# Target
y = merged_df['likes'].values

# Outlier clipping (99th percentile)
clip_value = np.percentile(y, 99)
y_clipped = np.clip(y, 0, clip_value)
print(f"\n[CLIP] Clipped likes at 99th percentile: {clip_value:.0f}")

# Log transform
y_log = np.log1p(y_clipped)

# Train-test split
np.random.seed(42)
train_idx, test_idx = train_test_split(
    np.arange(len(merged_df)),
    test_size=0.2,
    random_state=42
)

# =============================================================================
# EXPERIMENT 1: Baseline only (9 features)
# =============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 1: Baseline Features Only (9 features)")
print("=" * 80)

X_baseline = merged_df[baseline_cols].values
X_train_base = X_baseline[train_idx]
X_test_base = X_baseline[test_idx]
y_train = y_log[train_idx]
y_test_log = y_log[test_idx]
y_test_actual = y_clipped[test_idx]

# Quantile transform
qt = QuantileTransformer(output_distribution='normal', random_state=42)
X_train_qt = qt.fit_transform(X_train_base)
X_test_qt = qt.transform(X_test_base)

# Train ensemble
rf = RandomForestRegressor(n_estimators=250, max_depth=14, min_samples_split=3,
                           min_samples_leaf=2, random_state=42, n_jobs=-1)
hgb = HistGradientBoostingRegressor(max_iter=400, max_depth=14, learning_rate=0.05,
                                    min_samples_leaf=4, l2_regularization=0.1,
                                    random_state=42)

rf.fit(X_train_qt, y_train)
hgb.fit(X_train_qt, y_train)

# Predictions
pred_rf = rf.predict(X_test_qt)
pred_hgb = hgb.predict(X_test_qt)
pred_ensemble = 0.5 * pred_rf + 0.5 * pred_hgb
pred_actual = np.expm1(pred_ensemble)

mae_baseline = mean_absolute_error(y_test_actual, pred_actual)
r2_baseline = r2_score(y_test_actual, pred_actual)

print(f"\n[RESULTS]")
print(f"   MAE: {mae_baseline:.2f} likes")
print(f"   R²: {r2_baseline:.4f}")

# =============================================================================
# EXPERIMENT 2: Baseline + BERT PCA50 (59 features)
# =============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 2: Baseline + BERT PCA50 (59 features)")
print("=" * 80)

# PCA on BERT embeddings
print("\n[PCA] Reducing BERT 768 -> 50 dimensions...")
bert_array = merged_df[bert_cols].values
pca = PCA(n_components=50, random_state=42)
bert_pca = pca.fit_transform(bert_array)
variance_explained = pca.explained_variance_ratio_.sum()
print(f"   Variance explained: {variance_explained*100:.1f}%")

# Combine features
X_bert = np.concatenate([X_baseline, bert_pca], axis=1)
X_train_bert = X_bert[train_idx]
X_test_bert = X_bert[test_idx]

# Quantile transform
qt_bert = QuantileTransformer(output_distribution='normal', random_state=42)
X_train_qt_bert = qt_bert.fit_transform(X_train_bert)
X_test_qt_bert = qt_bert.transform(X_test_bert)

# Train ensemble
rf_bert = RandomForestRegressor(n_estimators=250, max_depth=14, min_samples_split=3,
                                min_samples_leaf=2, random_state=42, n_jobs=-1)
hgb_bert = HistGradientBoostingRegressor(max_iter=400, max_depth=14, learning_rate=0.05,
                                         min_samples_leaf=4, l2_regularization=0.1,
                                         random_state=42)

rf_bert.fit(X_train_qt_bert, y_train)
hgb_bert.fit(X_train_qt_bert, y_train)

# Predictions
pred_rf_bert = rf_bert.predict(X_test_qt_bert)
pred_hgb_bert = hgb_bert.predict(X_test_qt_bert)
pred_ensemble_bert = 0.5 * pred_rf_bert + 0.5 * pred_hgb_bert
pred_actual_bert = np.expm1(pred_ensemble_bert)

mae_bert = mean_absolute_error(y_test_actual, pred_actual_bert)
r2_bert = r2_score(y_test_actual, pred_actual_bert)

print(f"\n[RESULTS]")
print(f"   MAE: {mae_bert:.2f} likes")
print(f"   R²: {r2_bert:.4f}")
print(f"   Improvement vs baseline: {(mae_baseline - mae_bert) / mae_baseline * 100:.2f}%")

# =============================================================================
# EXPERIMENT 3: Baseline + BERT + NIMA (67 features)
# =============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 3: Baseline + BERT + NIMA (67 features)")
print("=" * 80)

# Combine features
nima_array = merged_df[nima_cols].values
X_full = np.concatenate([X_baseline, bert_pca, nima_array], axis=1)
X_train_full = X_full[train_idx]
X_test_full = X_full[test_idx]

# Quantile transform
qt_full = QuantileTransformer(output_distribution='normal', random_state=42)
X_train_qt_full = qt_full.fit_transform(X_train_full)
X_test_qt_full = qt_full.transform(X_test_full)

# Train ensemble
rf_full = RandomForestRegressor(n_estimators=250, max_depth=14, min_samples_split=3,
                                min_samples_leaf=2, random_state=42, n_jobs=-1)
hgb_full = HistGradientBoostingRegressor(max_iter=400, max_depth=14, learning_rate=0.05,
                                         min_samples_leaf=4, l2_regularization=0.1,
                                         random_state=42)

rf_full.fit(X_train_qt_full, y_train)
hgb_full.fit(X_train_qt_full, y_train)

# Predictions
pred_rf_full = rf_full.predict(X_test_qt_full)
pred_hgb_full = hgb_full.predict(X_test_qt_full)
pred_ensemble_full = 0.5 * pred_rf_full + 0.5 * pred_hgb_full
pred_actual_full = np.expm1(pred_ensemble_full)

mae_full = mean_absolute_error(y_test_actual, pred_actual_full)
r2_full = r2_score(y_test_actual, pred_actual_full)

print(f"\n[RESULTS]")
print(f"   MAE: {mae_full:.2f} likes")
print(f"   R²: {r2_full:.4f}")
print(f"   Improvement vs baseline: {(mae_baseline - mae_full) / mae_baseline * 100:.2f}%")
print(f"   Improvement vs BERT: {(mae_bert - mae_full) / mae_bert * 100:.2f}%")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("FINAL COMPARISON SUMMARY")
print("=" * 80)

print("\n[RESULTS TABLE]")
print(f"{'Model':<40} {'Features':<12} {'MAE':>10} {'R²':>10} {'vs Baseline'}")
print("-" * 80)
print(f"{'Baseline (9 features)':<40} {9:<12} {mae_baseline:>10.2f} {r2_baseline:>10.4f} {'-'}")
print(f"{'Baseline + BERT PCA50':<40} {59:<12} {mae_bert:>10.2f} {r2_bert:>10.4f} {(mae_baseline-mae_bert)/mae_baseline*100:>9.2f}%")
print(f"{'Baseline + BERT + NIMA':<40} {67:<12} {mae_full:>10.2f} {r2_full:>10.4f} {(mae_baseline-mae_full)/mae_baseline*100:>9.2f}%")

# Save results
results = pd.DataFrame([
    {'model': 'Baseline', 'features': 9, 'mae': mae_baseline, 'r2': r2_baseline},
    {'model': 'Baseline + BERT', 'features': 59, 'mae': mae_bert, 'r2': r2_bert},
    {'model': 'Baseline + BERT + NIMA', 'features': 67, 'mae': mae_full, 'r2': r2_full},
])

results.to_csv('experiments/multi_account_results.csv', index=False)
print(f"\n[SAVE] Results saved to: experiments/multi_account_results.csv")

# Comparison with 348-post dataset
print("\n[COMPARISON] With Previous 348-Post Dataset:")
print("   Previous best (348 posts, RFE 75 features): MAE = 135.21, R² = 0.4705")
print(f"   Current best ({len(baseline_df)} posts, 67 features): MAE = {mae_full:.2f}, R² = {r2_full:.4f}")

if mae_full < 135.21:
    improvement = (135.21 - mae_full) / 135.21 * 100
    print(f"   [SUCCESS] IMPROVEMENT: {improvement:.2f}% better MAE!")
else:
    degradation = (mae_full - 135.21) / 135.21 * 100
    print(f"   [WARNING] DEGRADATION: {degradation:.2f}% worse MAE")

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE!")
print("=" * 80 + "\n")
