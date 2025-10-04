#!/usr/bin/env python3
"""
PHASE 6: GBM ULTRA MODEL (FIXED DATA MERGE)
Properly deduplicates feature files before merging to fix the 188K row issue from Phase 5.1
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

print("="*90)
print(" "*20 + "PHASE 6: GBM ULTRA MODEL (FIXED)")
print(" "*15 + "Validating Phase 5.1 with Clean Data Merge")
print("="*90)
print()

# Load datasets
print("[LOAD] Loading datasets...")
df_main = pd.read_csv('multi_account_dataset.csv')
df_bert = pd.read_csv('data/processed/bert_embeddings_multi_account.csv')
df_aes = pd.read_csv('data/processed/aesthetic_features_multi_account.csv')

print(f"   Main: {len(df_main)} posts")
print(f"   BERT (raw): {len(df_bert)} posts")
print(f"   Aesthetic (raw): {len(df_aes)} posts")

# CRITICAL FIX: Deduplicate feature files BEFORE merging
print()
print("[DEDUPLICATE] Removing duplicates from feature files...")
df_bert_clean = df_bert.drop_duplicates(subset=['post_id', 'account'], keep='first')
df_aes_clean = df_aes.drop_duplicates(subset=['post_id', 'account'], keep='first')

print(f"   BERT: {len(df_bert)} -> {len(df_bert_clean)} ({len(df_bert) - len(df_bert_clean)} duplicates removed)")
print(f"   Aesthetic: {len(df_aes)} -> {len(df_aes_clean)} ({len(df_aes) - len(df_aes_clean)} duplicates removed)")

# Merge with clean data
print()
print("[MERGE] Merging datasets...")
df = df_main.merge(df_bert_clean, on=['post_id', 'account'], how='inner')
print(f"   After BERT merge: {len(df)} posts")

df = df.merge(df_aes_clean, on=['post_id', 'account'], how='inner')
print(f"   After aesthetic merge: {len(df)} posts")

# Prepare features
print()
print("[FEATURES] Preparing feature matrices...")

baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                 'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']
X_baseline = df[baseline_cols].values

bert_cols = [col for col in df.columns if col.startswith('bert_')]
X_bert = df[bert_cols].values
print(f"   BERT features: {X_bert.shape[1]}")

aes_cols = [col for col in df.columns if col.startswith('aesthetic_')]
X_aesthetic = df[aes_cols].values
print(f"   Aesthetic features: {X_aesthetic.shape[1]}")

y = df['likes'].values

# Train/test split
print()
print("[SPLIT] Creating train/test split...")
(X_baseline_train, X_baseline_test,
 X_bert_train, X_bert_test,
 X_aes_train, X_aes_test,
 y_train, y_test) = train_test_split(
    X_baseline, X_bert, X_aesthetic, y,
    test_size=0.2,
    random_state=42
)

print(f"   Train: {len(X_baseline_train)} posts")
print(f"   Test: {len(X_baseline_test)} posts")

# BERT PCA
print()
print("[PCA] Applying BERT PCA-150...")
pca_bert = PCA(n_components=150, random_state=42)
X_bert_pca_train = pca_bert.fit_transform(X_bert_train)
X_bert_pca_test = pca_bert.transform(X_bert_test)
print(f"   BERT: 768 -> 150 dims ({pca_bert.explained_variance_ratio_.sum():.1%} variance)")

# Combine features
X_train = np.hstack([X_baseline_train, X_bert_pca_train, X_aes_train])
X_test = np.hstack([X_baseline_test, X_bert_pca_test, X_aes_test])
print(f"   Total features: {X_train.shape[1]}")

# Preprocessing
print()
print("[PREPROCESS] Outlier clipping and scaling...")
clip_threshold = np.percentile(y_train, 99)
y_train_clipped = np.clip(y_train, 0, clip_threshold)
y_test_clipped = np.clip(y_test, 0, clip_threshold)
print(f"   Clipping at {clip_threshold:.0f} likes (99th percentile)")

y_train_log = np.log1p(y_train_clipped)
y_test_log = np.log1p(y_test_clipped)

scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train GBM
print()
print("="*90)
print(" "*25 + "TRAINING GBM MODEL")
print("="*90)
print()

gbm = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=3,
    subsample=0.8,
    random_state=42,
    verbose=0
)

gbm.fit(X_train_scaled, y_train_log)

# Predict
y_pred_log = gbm.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("[RESULTS]")
print(f"   MAE:  {mae:.2f} likes")
print(f"   RMSE: {rmse:.2f} likes")
print(f"   R2:   {r2:.4f}")

# Save model
print()
print("[SAVE] Saving production model...")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f'models/phase6_gbm_fixed_{timestamp}.pkl'

model_package = {
    'model': gbm,
    'scaler': scaler,
    'pca_bert': pca_bert,
    'baseline_features': baseline_cols,
    'aesthetic_features': aes_cols,
    'n_bert_pca': 150,
    'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2},
    'config': {
        'algorithm': 'GradientBoostingRegressor',
        'n_features': X_train.shape[1],
        'n_train': len(X_train),
        'n_test': len(X_test),
        'dataset_size': len(df)
    },
    'timestamp': timestamp
}

joblib.dump(model_package, model_filename)
import os
print(f"   Model: {model_filename}")
print(f"   Size: {os.path.getsize(model_filename)/(1024*1024):.1f} MB")

# Feature importance
print()
print("[IMPORTANCE] Top 15 features:")
feature_names = baseline_cols + [f'bert_pca_{i}' for i in range(150)] + [f'aes_{i}' for i in range(len(aes_cols))]
importances = gbm.feature_importances_
indices = np.argsort(importances)[::-1]

for i, idx in enumerate(indices[:15], 1):
    print(f"   {i:2d}. {feature_names[idx]:30s} {importances[idx]:.6f}")

# Compare to previous phases
print()
print("="*90)
print(" "*20 + "PERFORMANCE VS PREVIOUS PHASES")
print("="*90)
print()

phase5_mae = 27.23
improvement = ((phase5_mae - mae) / phase5_mae) * 100

print(f"[COMPARISON]")
print(f"   Phase 5 Ultra (BERT PCA-70):  MAE = {phase5_mae:.2f} likes")
print(f"   Phase 6 GBM (BERT PCA-150):   MAE = {mae:.2f} likes")
print(f"   Change:                       {improvement:+.1f}%")
print()

if mae < 10:
    print("[STATUS] EXCEPTIONAL - Sub-10 MAE achieved!")
elif mae < 20:
    print("[STATUS] EXCELLENT - Significant improvement!")
elif mae < phase5_mae:
    print("[STATUS] IMPROVED - Better than Phase 5!")
else:
    print("[STATUS] Phase 5.1's MAE=2.29 was on corrupted data (188K rows)")
    print("         This is the TRUE performance on clean 8.6K dataset")

print()
print("="*90)
print(" "*30 + "COMPLETE!")
print("="*90)
