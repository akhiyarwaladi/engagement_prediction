"""
Train Final Ultra Model - Phase 5 Ultra Champion
Configuration: Baseline (9 features) + BERT PCA-70 components

Based on Phase 5 ultra-optimization results:
- MAE: 14.09 likes (72.8% better than Phase 4 baseline)
- R²: 0.9562
- Configuration: RF 100% (no HGB), PCA=70, Baseline features
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

print("="*90)
print(" "*20 + "PHASE 5 ULTRA: FINAL CHAMPION MODEL TRAINING")
print(" "*25 + "Configuration: Baseline + BERT PCA-70")
print("="*90)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[LOAD] Loading datasets...")
df_main = pd.read_csv('multi_account_dataset.csv')
df_bert = pd.read_csv('data/processed/bert_embeddings_multi_account.csv')

print(f"   Main data: {len(df_main)} posts")
print(f"   BERT embeddings: {len(df_bert)} posts")

# Merge on post_id
df = df_main.merge(df_bert, on=['post_id', 'account'], how='inner')

print(f"\n[MERGE] Combined dataset: {len(df)} posts")

# ============================================================================
# PREPARE FEATURES
# ============================================================================
print("\n[FEATURES] Extracting features...")

# Baseline features (9)
baseline_features = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                     'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']

X_baseline = df[baseline_features].values

# BERT features (768-dim)
bert_cols = [col for col in df.columns if col.startswith('bert_dim_')]
X_bert_full = df[bert_cols].values

# Target
y = df['likes'].values

print(f"   Baseline features: {X_baseline.shape[1]}")
print(f"   BERT features: {X_bert_full.shape[1]}")
print(f"   Total posts: {len(df)}")

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================
print("\n[SPLIT] Creating train/test split...")

train_idx, test_idx = train_test_split(
    np.arange(len(df)),
    test_size=0.2,
    random_state=42
)

X_baseline_train = X_baseline[train_idx]
X_baseline_test = X_baseline[test_idx]
X_bert_train = X_bert_full[train_idx]
X_bert_test = X_bert_full[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

print(f"   Train: {len(train_idx)} posts")
print(f"   Test: {len(test_idx)} posts")

# ============================================================================
# PREPROCESSING
# ============================================================================
print("\n[PREPROCESS] Applying transformations...")

# Outlier clipping (99th percentile)
clip_value = np.percentile(y_train, 99)
y_train_clipped = np.clip(y_train, 0, clip_value)
y_test_clipped = np.clip(y_test, 0, clip_value)

# Log transform
y_train_log = np.log1p(y_train_clipped)
y_test_log = np.log1p(y_test_clipped)

print(f"   Outlier clipping: 99th percentile = {clip_value:.0f} likes")
print(f"   Log transform: applied")

# ============================================================================
# PCA ON BERT (70 COMPONENTS)
# ============================================================================
print("\n[PCA] Reducing BERT dimensions: 768 -> 70...")

pca_bert = PCA(n_components=70, random_state=42)
X_bert_pca_train = pca_bert.fit_transform(X_bert_train)
X_bert_pca_test = pca_bert.transform(X_bert_test)

variance_explained = pca_bert.explained_variance_ratio_.sum()
print(f"   Variance explained: {variance_explained:.1%}")

# ============================================================================
# COMBINE FEATURES
# ============================================================================
print("\n[COMBINE] Merging Baseline + BERT PCA...")

X_train_combined = np.hstack([X_baseline_train, X_bert_pca_train])
X_test_combined = np.hstack([X_baseline_test, X_bert_pca_test])

print(f"   Total features: {X_train_combined.shape[1]} (9 baseline + 70 BERT PCA)")

# ============================================================================
# SCALING
# ============================================================================
print("\n[SCALE] Quantile transformation to normal distribution...")

scaler = QuantileTransformer(output_distribution='normal', random_state=42)
X_train_scaled = scaler.fit_transform(X_train_combined)
X_test_scaled = scaler.transform(X_test_combined)

# ============================================================================
# TRAIN RANDOM FOREST (100% WEIGHT)
# ============================================================================
print("\n[TRAIN] Training Random Forest...")
print("   Configuration:")
print("      n_estimators: 250")
print("      max_depth: 14")
print("      min_samples_split: 3")
print("      min_samples_leaf: 2")
print("      max_features: sqrt")

rf = RandomForestRegressor(
    n_estimators=250,
    max_depth=14,
    min_samples_split=3,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

rf.fit(X_train_scaled, y_train_log)
print("   [OK] Training complete")

# ============================================================================
# PREDICT
# ============================================================================
print("\n[PREDICT] Evaluating on test set...")

pred_log = rf.predict(X_test_scaled)
pred = np.expm1(pred_log)

mae = mean_absolute_error(y_test_clipped, pred)
rmse = np.sqrt(mean_squared_error(y_test_clipped, pred))
r2 = r2_score(y_test_clipped, pred)

# Improvement over mean baseline
mean_baseline_mae = mean_absolute_error(y_test_clipped, [y_test_clipped.mean()] * len(y_test_clipped))
improvement = ((mean_baseline_mae - mae) / mean_baseline_mae) * 100

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "="*90)
print(" "*35 + "FINAL RESULTS")
print("="*90)

print(f"\n[PERFORMANCE]")
print(f"   MAE:  {mae:.2f} likes")
print(f"   RMSE: {rmse:.2f} likes")
print(f"   R²:   {r2:.4f}")
print(f"   Improvement vs mean: {improvement:.1f}%")

print(f"\n[COMPARISON TO PREVIOUS BEST]")
prev_best_mae = 51.82  # Phase 4 baseline champion
if mae < prev_best_mae:
    improvement_vs_prev = ((prev_best_mae - mae) / prev_best_mae) * 100
    print(f"   Previous best (Phase 4): MAE={prev_best_mae:.2f}")
    print(f"   Current (Phase 5 Ultra): MAE={mae:.2f}")
    print(f"   Improvement: {improvement_vs_prev:.1f}%")
    print(f"\n   [SUCCESS] NEW ULTRA CHAMPION!")
else:
    print(f"   Previous best: MAE={prev_best_mae:.2f}")
    print(f"   Current: MAE={mae:.2f}")
    print(f"   Status: Previous model still better")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n[FEATURE IMPORTANCE]")

feature_names = baseline_features + [f'bert_pca_{i}' for i in range(70)]
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print(f"\n[TOP 20] Most Important Features:")
for i in range(min(20, len(feature_names))):
    idx = indices[i]
    print(f"   {i+1:2d}. {feature_names[idx]:30s} {importances[idx]:.6f}")

# Group importance by type
baseline_importance = importances[:9].sum()
bert_importance = importances[9:].sum()

print(f"\n[IMPORTANCE BY TYPE]")
print(f"   Baseline features (9):     {baseline_importance:.1%}")
print(f"   BERT PCA features (70):    {bert_importance:.1%}")

# ============================================================================
# SAVE MODEL
# ============================================================================
print("\n[SAVE] Saving model...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"models/phase5_ultra_model.pkl"

model_package = {
    'model': rf,
    'pca_bert': pca_bert,
    'scaler': scaler,
    'baseline_features': baseline_features,
    'n_bert_pca': 70,
    'metrics': {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'improvement_vs_mean': improvement,
        'improvement_vs_prev_best': improvement_vs_prev if mae < prev_best_mae else 0
    },
    'config': {
        'model_type': 'RandomForest',
        'n_estimators': 250,
        'max_depth': 14,
        'bert_pca_components': 70,
        'bert_pca_variance': variance_explained,
        'scaler': 'QuantileTransformer(normal)',
        'features': X_train_combined.shape[1]
    },
    'dataset_info': {
        'total_posts': len(df),
        'train_posts': len(train_idx),
        'test_posts': len(test_idx),
        'outlier_clip_percentile': 99,
        'clip_value': clip_value
    },
    'timestamp': timestamp
}

joblib.dump(model_package, model_filename)
print(f"   Saved to: {model_filename}")

print("\n" + "="*90)
print(" "*25 + "PHASE 5 ULTRA MODEL COMPLETE!")
print("="*90)
