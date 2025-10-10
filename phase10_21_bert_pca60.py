#!/usr/bin/env python3
"""
PHASE 10.21: BERT PCA 60 COMPONENTS
Current best (Phase 10.19) uses 50 PCA components (88.4% variance)
Hypothesis: Increase to 60 components might recover lost BERT signal
Target: Beat MAE=43.74
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
print(" "*10 + "PHASE 10.21: BERT PCA 60 COMPONENTS (More Text Signal)")
print(" "*18 + "Target: Beat Phase 10.19 MAE=43.74")
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

# Visual (Phase 10.18 base + Phase 10.19 interactions)
metadata_base = ['file_size_kb', 'is_portrait', 'is_landscape', 'is_square']
df['resolution_log'] = np.log1p(df['resolution'])
df['aspect_ratio_sq'] = df['aspect_ratio'] ** 2

# Phase 10.19 interactions
df['aspect_x_logres'] = df['aspect_ratio'] * df['resolution_log']
df['filesize_x_logres'] = df['file_size_kb'] * df['resolution_log']
df['aspect_sq_x_logres'] = df['aspect_ratio_sq'] * df['resolution_log']

visual_features = metadata_base + ['aspect_ratio', 'resolution_log', 'aspect_ratio_sq',
                                    'aspect_x_logres', 'filesize_x_logres', 'aspect_sq_x_logres']

print(f"[STRATEGY] Phase 10.19 winner + BERT PCA 60 (was 50)")
print(f"   Baseline: {len(baseline_cols)} features")
print(f"   BERT: {len(bert_cols)} -> 60 PCA (was 50)")
print(f"   Visual: {len(visual_features)} features")
print()

X_baseline = df[baseline_cols].values
X_bert_full = df[bert_cols].values
X_visual = df[visual_features].values
y = df['likes'].values

# Split
train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)

X_baseline_train = X_baseline[train_idx]
X_baseline_test = X_baseline[test_idx]
X_bert_train = X_bert_full[train_idx]
X_bert_test = X_bert_full[test_idx]
X_visual_train = X_visual[train_idx]
X_visual_test = X_visual[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

# Preprocessing
clip_threshold = np.percentile(y_train, 99)
y_train_log = np.log1p(np.clip(y_train, 0, clip_threshold))
y_test_log = np.log1p(np.clip(y_test, 0, clip_threshold))

# PCA BERT with 60 components (was 50)
print("[PCA] Reducing BERT from 768 to 60 dimensions...")
pca_bert = PCA(n_components=60, random_state=42)
X_bert_pca_train = pca_bert.fit_transform(X_bert_train)
X_bert_pca_test = pca_bert.transform(X_bert_test)
variance_preserved = pca_bert.explained_variance_ratio_.sum()
print(f"   Variance preserved: {variance_preserved*100:.1f}% (was 88.4% with 50 components)")
print()

# Combine
X_train = np.hstack([X_baseline_train, X_bert_pca_train, X_visual_train])
X_test = np.hstack([X_baseline_test, X_bert_pca_test, X_visual_test])

print(f"[FEATURES] Total: {X_train.shape[1]} features (was 69 with PCA=50)")
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
print(f"[RESULT] Phase 10.21 BERT PCA 60: MAE={mae:.2f}, R2={r2:.4f}")
print("="*90)
print()

if mae < 43.74:
    improvement = (43.74 - mae) / 43.74 * 100
    print(f"[CHAMPION] NEW RECORD! Beat Phase 10.19 by {improvement:.2f}%!")
    print(f"   Phase 10.19: MAE=43.74")
    print(f"   Phase 10.21: MAE={mae:.2f}")
    print(f"   Improvement: {43.74 - mae:.2f} MAE points")
    print()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'models/phase10_21_bert_pca60_{timestamp}.pkl'
    model_package = {
        'phase': '10.21_bert_pca60',
        'mae': mae,
        'r2': r2,
        'features_count': X_train.shape[1],
        'bert_pca_components': 60,
        'bert_variance_preserved': variance_preserved,
        'visual_included': True,
        'text_included': True,
        'timestamp': timestamp
    }
    joblib.dump(model_package, model_filename)
    print(f"[SAVE] Model saved: {model_filename}")
    print()
else:
    print(f"[RESULT] Phase 10.19 remains champion")
    print(f"   Phase 10.19: MAE=43.74")
    print(f"   Phase 10.21: MAE={mae:.2f}")
    diff = mae - 43.74
    if diff > 0:
        print(f"   More PCA components added noise: +{diff:.2f} MAE")

print()
print("="*90)
print(" "*25 + "PHASE 10.21 COMPLETE!")
print("="*90)
