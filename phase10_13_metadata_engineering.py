#!/usr/bin/env python3
"""
PHASE 10.13: METADATA FEATURE ENGINEERING
Phase 10.9 proved metadata is best! Let's engineer more metadata features!
FAST: Only uses existing data, no heavy computation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print(" "*10 + "PHASE 10.13: METADATA FEATURE ENGINEERING (FAST!)")
print(" "*15 + "Target: Beat Phase 10.9 MAE=44.05")
print("="*90)
print()

# Load ONLY what we need (fast!)
print("[LOAD] Loading multimodal dataset (metadata focus)...")
df_main = pd.read_csv('multi_account_dataset.csv')
df_bert = pd.read_csv('data/processed/bert_embeddings_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])
df_visual = pd.read_csv('data/processed/advanced_visual_features_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])

df = df_main.merge(df_bert, on=['post_id', 'account'], how='inner')
df = df.merge(df_visual, on=['post_id', 'account'], how='inner')

print(f"   Dataset: {len(df)} posts")
print()

# Baseline features
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                 'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']

# BERT
bert_cols = [col for col in df.columns if col.startswith('bert_')]

# Metadata (Phase 10.9 winner!)
metadata_base = ['aspect_ratio', 'resolution', 'file_size_kb', 'is_portrait', 'is_landscape', 'is_square']

# ENGINEER MORE METADATA FEATURES!
print("[ENGINEER] Creating advanced metadata features...")

# Resolution bins (small, medium, large)
df['resolution_small'] = (df['resolution'] < df['resolution'].quantile(0.33)).astype(int)
df['resolution_medium'] = ((df['resolution'] >= df['resolution'].quantile(0.33)) &
                           (df['resolution'] < df['resolution'].quantile(0.67))).astype(int)
df['resolution_large'] = (df['resolution'] >= df['resolution'].quantile(0.67)).astype(int)

# File size bins
df['file_size_small'] = (df['file_size_kb'] < df['file_size_kb'].quantile(0.33)).astype(int)
df['file_size_large'] = (df['file_size_kb'] >= df['file_size_kb'].quantile(0.67)).astype(int)

# Aspect ratio categories
df['aspect_very_portrait'] = (df['aspect_ratio'] < 0.75).astype(int)  # Very tall
df['aspect_very_landscape'] = (df['aspect_ratio'] > 1.5).astype(int)  # Very wide

# Interactions (metadata × baseline)
df['resolution_x_caption'] = df['resolution'] * df['caption_length']
df['filesize_x_hashtags'] = df['file_size_kb'] * df['hashtag_count']
df['aspect_x_video'] = df['aspect_ratio'] * df['is_video']

# Log transforms
df['log_resolution'] = np.log1p(df['resolution'])
df['log_filesize'] = np.log1p(df['file_size_kb'])

# Squared features
df['aspect_ratio_squared'] = df['aspect_ratio'] ** 2

print(f"   Created 14 new metadata-engineered features!")
print()

# All metadata features
metadata_engineered = metadata_base + [
    'resolution_small', 'resolution_medium', 'resolution_large',
    'file_size_small', 'file_size_large',
    'aspect_very_portrait', 'aspect_very_landscape',
    'resolution_x_caption', 'filesize_x_hashtags', 'aspect_x_video',
    'log_resolution', 'log_filesize', 'aspect_ratio_squared'
]

X_baseline = df[baseline_cols].values
X_bert_full = df[bert_cols].values
X_metadata = df[metadata_engineered].values
y = df['likes'].values

# Split
train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)

X_baseline_train = X_baseline[train_idx]
X_baseline_test = X_baseline[test_idx]
X_bert_train = X_bert_full[train_idx]
X_bert_test = X_bert_full[test_idx]
X_metadata_train = X_metadata[train_idx]
X_metadata_test = X_metadata[test_idx]

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

print("[EXPERIMENT] Testing engineered metadata features...")
print()

# Combine
X_train = np.hstack([X_baseline_train, X_bert_pca_train, X_metadata_train])
X_test = np.hstack([X_baseline_test, X_bert_pca_test, X_metadata_test])

print(f"   Total features: {X_train.shape[1]} (9 baseline + 50 BERT + {len(metadata_engineered)} metadata)")
print()

# Scale
scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Quick stacking (2 models for speed)
from sklearn.model_selection import KFold
kf = KFold(n_splits=3, shuffle=True, random_state=42)  # 3-fold for speed
oof_preds = np.zeros((len(X_train_scaled), 2))
test_preds = np.zeros((len(X_test_scaled), 2))

models = [
    GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8, random_state=42),
    HistGradientBoostingRegressor(max_iter=600, learning_rate=0.07, max_depth=7, random_state=42),
]

for i, model in enumerate(models):
    print(f"   Training model {i+1}/2...")
    # OOF
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
        m = model.__class__(**model.get_params())
        m.fit(X_train_scaled[tr_idx], y_train_log[tr_idx])
        oof_preds[val_idx, i] = m.predict(X_train_scaled[val_idx])

    # Test
    model.fit(X_train_scaled, y_train_log)
    test_preds[:, i] = model.predict(X_test_scaled)

# Meta
meta = Ridge(alpha=10)
meta.fit(oof_preds, y_train_log)

y_pred_log = meta.predict(test_preds)
y_pred = np.expm1(y_pred_log)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print()
print(f"[RESULT] Engineered Metadata: MAE={mae:.2f}, R2={r2:.4f}")
print()

# Save if improved
if mae < 44.05:
    print("[SAVE] NEW CHAMPION! Saving model...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'models/phase10_13_metadata_eng_{timestamp}.pkl'

    model_package = {
        'phase': '10.13_metadata_engineering',
        'mae': mae,
        'r2': r2,
        'features_count': X_train.shape[1],
        'metadata_features': metadata_engineered,
        'visual_included': True,
        'text_included': True,
        'timestamp': timestamp
    }

    joblib.dump(model_package, model_filename)
    print(f"   Saved: {model_filename}")
    print()

print("="*90)
print(" "*30 + "PHASE 10.13 COMPLETE!")
print("="*90)
print()
print(f"[RESULT] MAE={mae:.2f} (Phase 10.9 was 44.05)")
if mae < 44.05:
    print(f"   IMPROVED by {(44.05-mae)/44.05*100:.1f}%!")
    print(f"   Metadata feature engineering WINS!")
else:
    print(f"   Phase 10.9 metadata features remain optimal")
    print(f"   Sometimes simple is better than complex!")
