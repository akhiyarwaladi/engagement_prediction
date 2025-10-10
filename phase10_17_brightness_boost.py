#!/usr/bin/env python3
"""
PHASE 10.17: BRIGHTNESS BOOST
Phase 10.9 showed color features (esp. brightness) helped (MAE=44.72)
Phase 10.16 proved log(resolution) works (MAE=43.92)
Now combine: Metadata + log(resolution) + brightness + saturation
Target: Beat MAE=43.92
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
print(" "*10 + "PHASE 10.17: BRIGHTNESS BOOST (Metadata + Color Selective)")
print(" "*18 + "Target: Beat Phase 10.16 MAE=43.92")
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

# Enhanced metadata + selective color
metadata_base = ['aspect_ratio', 'file_size_kb', 'is_portrait', 'is_landscape', 'is_square']

# Enhancements
df['resolution_log'] = np.log1p(df['resolution'])  # Phase 10.16 winner
df['brightness_norm'] = df['brightness'] / 255.0  # Normalize brightness to [0,1]
df['saturation_norm'] = df['saturation'] / 255.0  # Normalize saturation to [0,1]

enhanced_features = metadata_base + ['resolution_log', 'brightness_norm', 'saturation_norm']

print("[STRATEGY] Phase 10.16 + brightness + saturation (normalized)")
print(f"   Baseline: {len(baseline_cols)} features")
print(f"   BERT: {len(bert_cols)} -> 50 PCA")
print(f"   Enhanced: {len(enhanced_features)} features (metadata + log_res + color)")
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
print(f"   - Baseline: {len(baseline_cols)}")
print(f"   - BERT PCA: 50")
print(f"   - Enhanced: {len(enhanced_features)}")
print()

# Scale
scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Stacking (5-fold, 4 models)
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
    # OOF
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
        m = model.__class__(**model.get_params())
        m.fit(X_train_scaled[tr_idx], y_train_log[tr_idx])
        oof_preds[val_idx, i] = m.predict(X_train_scaled[val_idx])

    # Test
    model.fit(X_train_scaled, y_train_log)
    test_preds[:, i] = model.predict(X_test_scaled)

print()

# Meta-learner
meta = Ridge(alpha=10)
meta.fit(oof_preds, y_train_log)

y_pred_log = meta.predict(test_preds)
y_pred = np.expm1(y_pred_log)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("="*90)
print(f"[RESULT] Phase 10.17 Brightness Boost: MAE={mae:.2f}, R2={r2:.4f}")
print("="*90)
print()

if mae < 43.92:
    improvement = (43.92 - mae) / 43.92 * 100
    print(f"[CHAMPION] NEW RECORD! Beat Phase 10.16 by {improvement:.2f}%!")
    print(f"   Phase 10.16: MAE=43.92")
    print(f"   Phase 10.17: MAE={mae:.2f}")
    print(f"   Improvement: {43.92 - mae:.2f} MAE points")
    print()

    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'models/phase10_17_brightness_boost_{timestamp}.pkl'

    model_package = {
        'phase': '10.17_brightness_boost',
        'mae': mae,
        'r2': r2,
        'features_count': X_train.shape[1],
        'enhancements': ['log_resolution', 'brightness_norm', 'saturation_norm'],
        'visual_included': True,
        'text_included': True,
        'timestamp': timestamp
    }

    joblib.dump(model_package, model_filename)
    print(f"[SAVE] Model saved: {model_filename}")
    print()
else:
    print(f"[RESULT] Phase 10.16 remains champion")
    print(f"   Phase 10.16: MAE=43.92")
    print(f"   Phase 10.17: MAE={mae:.2f}")
    diff = mae - 43.92
    if diff > 0:
        print(f"   Color features added noise: +{diff:.2f} MAE (worse)")
    else:
        print(f"   Matched performance!")

print()
print("="*90)
print(" "*25 + "PHASE 10.17 COMPLETE!")
print("="*90)
