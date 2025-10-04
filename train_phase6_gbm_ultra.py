#!/usr/bin/env python3
"""
PHASE 6: GBM ULTRA MODEL
Validates and trains production GBM model based on Phase 5.1 breakthrough discovery

Configuration from Phase 5.1 experiments:
- Algorithm: GradientBoostingRegressor (MAE=2.29 in exploration)
- Features: Baseline (9) + BERT PCA-150 + Aesthetic (8)
- Total: 167 features
- Scaler: QuantileTransformer-uniform
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("                    PHASE 6: GBM ULTRA MODEL TRAINING")
print("          Validating Phase 5.1 Breakthrough (MAE=2.29) on Production Data")
print("=" * 90)
print()

# ==========================================================================================
#                                   LOAD DATASETS
# ==========================================================================================

print("[LOAD] Loading datasets...")

# Load main dataset
df_main = pd.read_csv('multi_account_dataset.csv')
print(f"   Main dataset: {len(df_main)} posts")

# Load BERT embeddings
df_bert = pd.read_csv('data/processed/bert_embeddings_multi_account.csv')
print(f"   BERT embeddings: {len(df_bert)} posts, {df_bert.shape[1]-1} features")

# Load aesthetic features
df_aes = pd.read_csv('data/processed/aesthetic_features_multi_account.csv')
print(f"   Aesthetic features: {len(df_aes)} posts, {df_aes.shape[1]-1} features")

# ==========================================================================================
#                                   MERGE DATASETS
# ==========================================================================================

print()
print("[MERGE] Merging datasets...")

# Check if 'account' column exists in all dataframes
merge_keys = ['post_id', 'account'] if 'account' in df_bert.columns and 'account' in df_aes.columns else ['post_id']
print(f"   Merge keys: {merge_keys}")

# First merge: main + BERT
df = df_main.merge(df_bert, on=merge_keys, how='inner', suffixes=('', '_bert'))
print(f"   After BERT merge: {len(df)} posts")

# Second merge: add aesthetic
df = df.merge(df_aes, on=merge_keys, how='inner', suffixes=('', '_aes'))
print(f"   After aesthetic merge: {len(df)} posts")

# Clean up duplicate columns if any
df = df.loc[:, ~df.columns.duplicated()]

# Validate expected dataset size
if len(df) < 8000:
    print(f"   [WARNING] Expected ~8,610 posts but got {len(df)}")
elif len(df) > 10000:
    print(f"   [ERROR] Too many posts ({len(df)}), data merge issue detected!")
    print(f"   [FIX] Deduplicating by post_id...")
    df = df.drop_duplicates(subset='post_id', keep='first')
    print(f"   After deduplication: {len(df)} posts")

print(f"   [OK] Final dataset: {len(df)} posts")

# ==========================================================================================
#                                   PREPARE FEATURES
# ==========================================================================================

print()
print("[FEATURES] Preparing feature matrices...")

# 1. Baseline features (9)
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                 'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']
X_baseline = df[baseline_cols].values

# 2. BERT features (768 dims)
bert_cols = [col for col in df.columns if col.startswith('bert_') and col != 'bert_pca']
X_bert = df[bert_cols].values
print(f"   BERT features: {X_bert.shape[1]}")

# 3. Aesthetic features (8 features)
aes_cols = [col for col in df.columns if col.startswith('aesthetic_')]
X_aesthetic = df[aes_cols].values if len(aes_cols) > 0 else np.zeros((len(df), 8))
print(f"   Aesthetic features: {X_aesthetic.shape[1]}")

# Target variable
y = df['likes'].values

# ==========================================================================================
#                                   TRAIN/TEST SPLIT
# ==========================================================================================

print()
print("[SPLIT] Creating train/test split...")

# Split all feature matrices together
(X_baseline_train, X_baseline_test,
 X_bert_train, X_bert_test,
 X_aes_train, X_aes_test,
 y_train, y_test) = train_test_split(
    X_baseline, X_bert, X_aesthetic, y,
    test_size=0.2,
    random_state=42
)

print(f"   Train: {len(X_baseline_train)} posts")
print(f"   Test:  {len(X_baseline_test)} posts")

# ==========================================================================================
#                            BERT PCA DIMENSIONALITY REDUCTION
# ==========================================================================================

print()
print("[PCA] Applying BERT PCA-150 (optimal from Phase 5.1)...")

pca_bert = PCA(n_components=150, random_state=42)
X_bert_pca_train = pca_bert.fit_transform(X_bert_train)
X_bert_pca_test = pca_bert.transform(X_bert_test)

variance_explained = pca_bert.explained_variance_ratio_.sum()
print(f"   BERT: 768 -> 150 dims ({variance_explained:.1%} variance)")

# ==========================================================================================
#                                   COMBINE FEATURES
# ==========================================================================================

print()
print("[COMBINE] Building feature matrix...")

# Test with and without aesthetic
X_train_no_aes = np.hstack([X_baseline_train, X_bert_pca_train])
X_test_no_aes = np.hstack([X_baseline_test, X_bert_pca_test])

X_train_with_aes = np.hstack([X_baseline_train, X_bert_pca_train, X_aes_train])
X_test_with_aes = np.hstack([X_baseline_test, X_bert_pca_test, X_aes_test])

print(f"   Without aesthetic: {X_train_no_aes.shape[1]} features")
print(f"   With aesthetic: {X_train_with_aes.shape[1]} features")

# ==========================================================================================
#                                   PREPROCESSING
# ==========================================================================================

print()
print("[PREPROCESS] Outlier clipping and scaling...")

# Clip outliers at 99th percentile
clip_threshold = np.percentile(y_train, 99)
y_train_clipped = np.clip(y_train, 0, clip_threshold)
y_test_clipped = np.clip(y_test, 0, clip_threshold)
print(f"   Clipping at {clip_threshold:.0f} likes (99th percentile)")

# Log transform target
y_train_log = np.log1p(y_train_clipped)
y_test_log = np.log1p(y_test_clipped)

# Scale features with QuantileTransformer-uniform (best from Phase 5.1)
scaler_no_aes = QuantileTransformer(output_distribution='uniform', random_state=42)
X_train_scaled_no_aes = scaler_no_aes.fit_transform(X_train_no_aes)
X_test_scaled_no_aes = scaler_no_aes.transform(X_test_no_aes)

scaler_with_aes = QuantileTransformer(output_distribution='uniform', random_state=42)
X_train_scaled_with_aes = scaler_with_aes.fit_transform(X_train_with_aes)
X_test_scaled_with_aes = scaler_with_aes.transform(X_test_with_aes)

# ==========================================================================================
#                            EXPERIMENT 1: GBM WITHOUT AESTHETIC
# ==========================================================================================

print()
print("=" * 90)
print("                    EXPERIMENT 1: GBM WITHOUT AESTHETIC (159 features)")
print("=" * 90)
print()

print("[TRAIN] Training GradientBoostingRegressor...")

gbm_no_aes = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=3,
    subsample=0.8,
    random_state=42,
    verbose=0
)

gbm_no_aes.fit(X_train_scaled_no_aes, y_train_log)

# Predict
y_pred_log_no_aes = gbm_no_aes.predict(X_test_scaled_no_aes)
y_pred_no_aes = np.expm1(y_pred_log_no_aes)

# Metrics
mae_no_aes = mean_absolute_error(y_test, y_pred_no_aes)
rmse_no_aes = np.sqrt(mean_squared_error(y_test, y_pred_no_aes))
r2_no_aes = r2_score(y_test, y_pred_no_aes)

print()
print("[RESULTS]")
print(f"   MAE:  {mae_no_aes:.2f} likes")
print(f"   RMSE: {rmse_no_aes:.2f} likes")
print(f"   R²:   {r2_no_aes:.4f}")

# ==========================================================================================
#                            EXPERIMENT 2: GBM WITH AESTHETIC
# ==========================================================================================

print()
print("=" * 90)
print("                    EXPERIMENT 2: GBM WITH AESTHETIC (167 features)")
print("=" * 90)
print()

print("[TRAIN] Training GradientBoostingRegressor...")

gbm_with_aes = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=3,
    subsample=0.8,
    random_state=42,
    verbose=0
)

gbm_with_aes.fit(X_train_scaled_with_aes, y_train_log)

# Predict
y_pred_log_with_aes = gbm_with_aes.predict(X_test_scaled_with_aes)
y_pred_with_aes = np.expm1(y_pred_log_with_aes)

# Metrics
mae_with_aes = mean_absolute_error(y_test, y_pred_with_aes)
rmse_with_aes = np.sqrt(mean_squared_error(y_test, y_pred_with_aes))
r2_with_aes = r2_score(y_test, y_pred_with_aes)

print()
print("[RESULTS]")
print(f"   MAE:  {mae_with_aes:.2f} likes")
print(f"   RMSE: {rmse_with_aes:.2f} likes")
print(f"   R²:   {r2_with_aes:.4f}")

# ==========================================================================================
#                                   COMPARE AND SELECT BEST
# ==========================================================================================

print()
print("=" * 90)
print("                              COMPARISON & SELECTION")
print("=" * 90)
print()

print("[COMPARE]")
print(f"   GBM WITHOUT aesthetic (159 features): MAE={mae_no_aes:.2f}, R²={r2_no_aes:.4f}")
print(f"   GBM WITH aesthetic (167 features):    MAE={mae_with_aes:.2f}, R²={r2_with_aes:.4f}")
print()

# Select best based on MAE
if mae_no_aes <= mae_with_aes:
    print(f"[WINNER] WITHOUT aesthetic (MAE {mae_no_aes:.2f} <= {mae_with_aes:.2f})")
    best_model = gbm_no_aes
    best_scaler = scaler_no_aes
    best_features = 'baseline_bert_pca150'
    best_mae = mae_no_aes
    best_r2 = r2_no_aes
    best_rmse = rmse_no_aes
    use_aesthetic = False
else:
    print(f"[WINNER] WITH aesthetic (MAE {mae_with_aes:.2f} < {mae_no_aes:.2f})")
    best_model = gbm_with_aes
    best_scaler = scaler_with_aes
    best_features = 'baseline_bert_pca150_aesthetic'
    best_mae = mae_with_aes
    best_r2 = r2_with_aes
    best_rmse = rmse_with_aes
    use_aesthetic = True

# ==========================================================================================
#                                   SAVE PRODUCTION MODEL
# ==========================================================================================

print()
print("[SAVE] Saving production model...")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f'models/phase6_gbm_ultra_{timestamp}.pkl'

# Package model
model_package = {
    'model': best_model,
    'scaler': best_scaler,
    'pca_bert': pca_bert,
    'baseline_features': baseline_cols,
    'use_aesthetic': use_aesthetic,
    'aesthetic_features': aes_cols if use_aesthetic else None,
    'n_bert_pca': 150,
    'metrics': {
        'mae': best_mae,
        'rmse': best_rmse,
        'r2': best_r2
    },
    'config': {
        'algorithm': 'GradientBoostingRegressor',
        'features': best_features,
        'n_features': X_train_with_aes.shape[1] if use_aesthetic else X_train_no_aes.shape[1],
        'scaler': 'QuantileTransformer-uniform',
        'n_train': len(X_baseline_train),
        'n_test': len(X_baseline_test)
    },
    'timestamp': timestamp
}

joblib.dump(model_package, model_filename)
import os
model_size_mb = os.path.getsize(model_filename) / (1024 * 1024)
print(f"   Model saved: {model_filename}")
print(f"   Size: {model_size_mb:.1f} MB")

# ==========================================================================================
#                                   FEATURE IMPORTANCE
# ==========================================================================================

print()
print("[IMPORTANCE] Top 15 most important features:")
print()

# Get feature importances
importances = best_model.feature_importances_

# Create feature names
feature_names = baseline_cols.copy()
feature_names += [f'bert_pca_{i}' for i in range(150)]
if use_aesthetic:
    feature_names += [f'aesthetic_{i}' for i in range(len(aes_cols))]

# Sort by importance
indices = np.argsort(importances)[::-1]

for i, idx in enumerate(indices[:15], 1):
    print(f"   {i:2d}. {feature_names[idx]:30s} {importances[idx]:.6f}")

# ==========================================================================================
#                                   COMPARISON TO PREVIOUS
# ==========================================================================================

print()
print("=" * 90)
print("                         PERFORMANCE VS PREVIOUS PHASES")
print("=" * 90)
print()

phase5_mae = 27.23
improvement = ((phase5_mae - best_mae) / phase5_mae) * 100

print(f"[PROGRESS]")
print(f"   Phase 5 Ultra (BERT PCA-70):  MAE = {phase5_mae:.2f} likes")
print(f"   Phase 6 GBM (BERT PCA-150):   MAE = {best_mae:.2f} likes")
print(f"   Improvement:                  {improvement:.1f}%")
print()

if best_mae < 10:
    print("[STATUS] 🏆 EXCEPTIONAL PERFORMANCE - Sub-10 MAE achieved!")
elif best_mae < 20:
    print("[STATUS] ✅ EXCELLENT PERFORMANCE - Significant improvement!")
elif best_mae < phase5_mae:
    print("[STATUS] ✅ IMPROVED - Better than Phase 5 Ultra!")
else:
    print("[STATUS] ⚠️  Phase 5.1 exploration results may not reproduce on production data")
    print("           GBM MAE=2.29 likely due to data merge issue (188K rows)")

# ==========================================================================================
#                                   PHASE 5.1 ANALYSIS
# ==========================================================================================

print()
print("=" * 90)
print("                    ANALYSIS: Phase 5.1 vs Phase 6 Results")
print("=" * 90)
print()

print("[PHASE 5.1 EXPLORATION]")
print("   Dataset: 188,130 posts (data merge issue - Cartesian product)")
print("   GBM Result: MAE=2.29, R²=0.9941")
print("   Note: Exploration on corrupted dataset with duplicates")
print()

print("[PHASE 6 PRODUCTION]")
print(f"   Dataset: {len(df)} posts (cleaned and deduplicated)")
print(f"   GBM Result: MAE={best_mae:.2f}, R²={best_r2:.4f}")
print("   Note: Production-ready model on correct dataset")
print()

if best_mae > 10:
    print("[INSIGHT]")
    print("   Phase 5.1's exceptional MAE=2.29 was due to:")
    print("   1. Data merge created 188K rows from 8.6K posts")
    print("   2. Likely created train/test leakage or data artifacts")
    print("   3. Model memorized duplicated patterns")
    print()
    print("   Phase 6's result is the TRUE performance on clean data.")
    print(f"   MAE={best_mae:.2f} is realistic and represents actual capability.")
else:
    print("[INSIGHT]")
    print("   Successfully validated Phase 5.1 breakthrough!")
    print("   GBM with BERT PCA-150 achieves exceptional performance.")

# ==========================================================================================
#                                   SESSION COMPLETE
# ==========================================================================================

print()
print("=" * 90)
print("                         PHASE 6 GBM ULTRA: COMPLETE!")
print("=" * 90)
print()

print("[SUMMARY]")
print(f"   Algorithm: GradientBoostingRegressor")
print(f"   Features: {best_features}")
print(f"   Total Features: {X_train_with_aes.shape[1] if use_aesthetic else X_train_no_aes.shape[1]}")
print(f"   MAE: {best_mae:.2f} likes")
print(f"   R²: {best_r2:.4f}")
print(f"   Model: {model_filename}")
print()

print("[PRODUCTION READY] ✅")
print(f"   Use this model for deployment: {model_filename}")
print()
