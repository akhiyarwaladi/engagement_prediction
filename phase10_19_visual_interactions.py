#!/usr/bin/env python3
"""
PHASE 10.19: VISUAL FEATURE INTERACTIONS
Hypothesis: Interaction between visual metadata might capture composition patterns
Test: aspect × resolution, file_size × resolution interactions
Target: Beat MAE=43.89
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print(" "*6 + "PHASE 10.19: VISUAL FEATURE INTERACTIONS (Metadata Cross-Effects)")
print(" "*18 + "Target: Beat Phase 10.18 MAE=43.89")
print("="*90)
print()

# Load data
print("[LOAD] Loading multimodal dataset...")
df_main = pd.read_csv('multi_account_dataset.csv')
df_bert = pd.read_csv('data/processed/bert_embeddings_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])
df_visual = pd.read_csv('data/processed/advanced_visual_features_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])

df = df_main.merge(df_bert, on=['post_id', 'account'], how='inner')
df = df.merge(df_visual, on=['post_id', 'account'], how='inner')

print(f"   Dataset: {len(df)} posts")
print()

# Baseline
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                 'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']

# BERT
bert_cols = [col for col in df.columns if col.startswith('bert_')]

# Base metadata (from Phase 10.18 winner)
metadata_base = ['file_size_kb', 'is_portrait', 'is_landscape', 'is_square']

# Phase 10.18 enhancements
df['resolution_log'] = np.log1p(df['resolution'])
df['aspect_ratio_sq'] = df['aspect_ratio'] ** 2

# NEW: Visual Interactions
print("[ENGINEER] Creating visual interaction features...")

# Interaction 1: aspect × log_resolution (composition × quality)
df['aspect_x_logres'] = df['aspect_ratio'] * df['resolution_log']

# Interaction 2: file_size × log_resolution (quality consistency)
df['filesize_x_logres'] = df['file_size_kb'] * df['resolution_log']

# Interaction 3: aspect² × log_resolution (non-linear composition effect)
df['aspect_sq_x_logres'] = df['aspect_ratio_sq'] * df['resolution_log']

interaction_features = ['aspect_x_logres', 'filesize_x_logres', 'aspect_sq_x_logres']

print(f"   Created 3 visual interaction features:")
print(f"     - aspect_ratio × log(resolution)")
print(f"     - file_size × log(resolution)")
print(f"     - aspect² × log(resolution)")
print()

# Combined features
enhanced_features = metadata_base + ['aspect_ratio', 'resolution_log', 'aspect_ratio_sq'] + interaction_features

print(f"[STRATEGY] Phase 10.18 + visual interactions")
print(f"   Baseline: {len(baseline_cols)} features")
print(f"   BERT: {len(bert_cols)} -> 50 PCA")
print(f"   Enhanced: {len(enhanced_features)} features")
print()

X_baseline = df[baseline_cols].values
X_bert_full = df[bert_cols].values
X_enhanced = df[enhanced_features].values
y = df['likes'].values

# Split
train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)

X_baseline_train = X_baseline[train_idx]
X_baseline_test = X_baseline[test_idx]
X_bert_train = X_bert_full[train_idx]
X_bert_test = X_bert_full[test_idx]
X_enhanced_train = X_enhanced[train_idx]
X_enhanced_test = X_enhanced[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

# Preprocessing
clip_threshold = np.percentile(y_train, 99)
y_train_log = np.log1p(np.clip(y_train, 0, clip_threshold))
y_test_log = np.log1p(np.clip(y_test, 0, clip_threshold))

# PCA BERT
pca_bert = PCA(n_components=50, random_state=42)
X_bert_pca_train = pca_bert.fit_transform(X_bert_train)
X_bert_pca_test = pca_bert.transform(X_bert_test)

# Combine
X_train = np.hstack([X_baseline_train, X_bert_pca_train, X_enhanced_train])
X_test = np.hstack([X_baseline_test, X_bert_pca_test, X_enhanced_test])

print(f"[FEATURES] Total: {X_train.shape[1]} features")
print()

# Scale
scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Stacking
print("[MODEL] Training 4-model stacking ensemble (5-fold CV)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X_train_scaled), 4))
test_preds = np.zeros((len(X_test_scaled), 4))

models = [
    GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8, random_state=42),
    HistGradientBoostingRegressor(max_iter=600, learning_rate=0.07, max_depth=7, random_state=42),
    RandomForestRegressor(n_estimators=300, max_depth=16, random_state=42, n_jobs=-1),
    ExtraTreesRegressor(n_estimators=300, max_depth=16, random_state=42, n_jobs=-1),
]

for i, model in enumerate(models):
    print(f"   Model {i+1}/4: {model.__class__.__name__}")
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
        m = model.__class__(**model.get_params())
        m.fit(X_train_scaled[tr_idx], y_train_log[tr_idx])
        oof_preds[val_idx, i] = m.predict(X_train_scaled[val_idx])
    model.fit(X_train_scaled, y_train_log)
    test_preds[:, i] = model.predict(X_test_scaled)

print()

# Meta
meta = Ridge(alpha=10)
meta.fit(oof_preds, y_train_log)

y_pred_log = meta.predict(test_preds)
y_pred = np.expm1(y_pred_log)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("="*90)
print(f"[RESULT] Phase 10.19 Visual Interactions: MAE={mae:.2f}, R2={r2:.4f}")
print("="*90)
print()

if mae < 43.89:
    improvement = (43.89 - mae) / 43.89 * 100
    print(f"[CHAMPION] NEW RECORD! Beat Phase 10.18 by {improvement:.2f}%!")
    print(f"   Phase 10.18: MAE=43.89")
    print(f"   Phase 10.19: MAE={mae:.2f}")
    print(f"   Improvement: {43.89 - mae:.2f} MAE points")
    print()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'models/phase10_19_visual_interactions_{timestamp}.pkl'
    model_package = {
        'phase': '10.19_visual_interactions',
        'mae': mae,
        'r2': r2,
        'features_count': X_train.shape[1],
        'interactions': interaction_features,
        'visual_included': True,
        'text_included': True,
        'timestamp': timestamp
    }
    joblib.dump(model_package, model_filename)
    print(f"[SAVE] Model saved: {model_filename}")
    print()
else:
    print(f"[RESULT] Phase 10.18 remains champion")
    print(f"   Phase 10.18: MAE=43.89")
    print(f"   Phase 10.19: MAE={mae:.2f}")
    diff = mae - 43.89
    if diff > 0:
        print(f"   Interactions added noise: +{diff:.2f} MAE")

print()
print("="*90)
print(" "*25 + "PHASE 10.19 COMPLETE!")
print("="*90)
