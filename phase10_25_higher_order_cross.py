#!/usr/bin/env python3
"""
PHASE 10.25: HIGHER-ORDER CROSS INTERACTIONS
Phase 10.23 proved text×visual cross works (MAE=43.64)
Hypothesis: (caption × aspect)², (hashtag × resolution)² capture non-linear multimodal effects
Target: Beat MAE=43.64
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
print(" "*5 + "PHASE 10.25: HIGHER-ORDER CROSS INTERACTIONS (Non-linear Multimodal)")
print(" "*18 + "Target: Beat Phase 10.23 MAE=43.64")
print("="*90)
print()

df_main = pd.read_csv('multi_account_dataset.csv')
df_bert = pd.read_csv('data/processed/bert_embeddings_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])
df_visual = pd.read_csv('data/processed/advanced_visual_features_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])
df = df_main.merge(df_bert, on=['post_id', 'account'], how='inner').merge(df_visual, on=['post_id', 'account'], how='inner')

print(f"[LOAD] Dataset: {len(df)} posts")
print()

baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                 'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']
bert_cols = [col for col in df.columns if col.startswith('bert_')]

# Phase 10.23 base features
metadata_base = ['file_size_kb', 'is_portrait', 'is_landscape', 'is_square']
df['resolution_log'] = np.log1p(df['resolution'])
df['aspect_ratio_sq'] = df['aspect_ratio'] ** 2
df['aspect_x_logres'] = df['aspect_ratio'] * df['resolution_log']
df['filesize_x_logres'] = df['file_size_kb'] * df['resolution_log']
df['aspect_sq_x_logres'] = df['aspect_ratio_sq'] * df['resolution_log']

# Text-visual cross interactions (Phase 10.23)
df['caption_x_aspect'] = df['caption_length'] * df['aspect_ratio']
df['caption_x_logres'] = df['caption_length'] * df['resolution_log']
df['hashtag_x_logres'] = df['hashtag_count'] * df['resolution_log']
df['word_x_filesize'] = df['word_count'] * df['file_size_kb']
df['caption_x_filesize'] = df['caption_length'] * df['file_size_kb']

# NEW: Higher-order (squared) cross interactions
print("[ENGINEER] Creating higher-order cross interactions...")
df['caption_x_aspect_sq'] = df['caption_x_aspect'] ** 2
df['caption_x_logres_sq'] = df['caption_x_logres'] ** 2
df['hashtag_x_logres_sq'] = df['hashtag_x_logres'] ** 2
df['word_x_filesize_sq'] = df['word_x_filesize'] ** 2

# Product of different cross terms (multimodal synergy)
df['caption_aspect_x_hashtag_res'] = df['caption_x_aspect'] * df['hashtag_x_logres']
df['word_filesize_x_caption_res'] = df['word_x_filesize'] * df['caption_x_logres']

higher_order = ['caption_x_aspect_sq', 'caption_x_logres_sq', 'hashtag_x_logres_sq',
                'word_x_filesize_sq', 'caption_aspect_x_hashtag_res', 'word_filesize_x_caption_res']

print(f"   Created {len(higher_order)} higher-order cross features")
print()

cross_interactions = ['caption_x_aspect', 'caption_x_logres', 'hashtag_x_logres',
                      'word_x_filesize', 'caption_x_filesize']

visual_features = (metadata_base + ['aspect_ratio', 'resolution_log', 'aspect_ratio_sq',
                   'aspect_x_logres', 'filesize_x_logres', 'aspect_sq_x_logres'] +
                   cross_interactions + higher_order)

print(f"[STRATEGY] Phase 10.23 + higher-order cross interactions")
print(f"   Baseline: {len(baseline_cols)}, BERT PCA: 60, Visual+Cross+Higher: {len(visual_features)}")
print()

X_baseline = df[baseline_cols].values
X_bert_full = df[bert_cols].values
X_visual = df[visual_features].values
y = df['likes'].values

train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
X_baseline_train, X_baseline_test = X_baseline[train_idx], X_baseline[test_idx]
X_bert_train, X_bert_test = X_bert_full[train_idx], X_bert_full[test_idx]
X_visual_train, X_visual_test = X_visual[train_idx], X_visual[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

clip_threshold = np.percentile(y_train, 99)
y_train_log = np.log1p(np.clip(y_train, 0, clip_threshold))
y_test_log = np.log1p(np.clip(y_test, 0, clip_threshold))

pca_bert = PCA(n_components=60, random_state=42)
X_bert_pca_train = pca_bert.fit_transform(X_bert_train)
X_bert_pca_test = pca_bert.transform(X_bert_test)

X_train = np.hstack([X_baseline_train, X_bert_pca_train, X_visual_train])
X_test = np.hstack([X_baseline_test, X_bert_pca_test, X_visual_test])

scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("[MODEL] Training ensemble...")
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
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
        m = model.__class__(**model.get_params())
        m.fit(X_train_scaled[tr_idx], y_train_log[tr_idx])
        oof_preds[val_idx, i] = m.predict(X_train_scaled[val_idx])
    model.fit(X_train_scaled, y_train_log)
    test_preds[:, i] = model.predict(X_test_scaled)

meta = Ridge(alpha=10)
meta.fit(oof_preds, y_train_log)
y_pred = np.expm1(meta.predict(test_preds))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("="*90)
print(f"[RESULT] Phase 10.25: MAE={mae:.2f}, R2={r2:.4f}")
print("="*90)

if mae < 43.64:
    print(f"[CHAMPION] NEW RECORD! Beat Phase 10.23 by {((43.64-mae)/43.64*100):.2f}%!")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    joblib.dump({'phase': '10.25', 'mae': mae, 'r2': r2, 'higher_order': higher_order, 'timestamp': timestamp},
                f'models/phase10_25_higher_order_cross_{timestamp}.pkl')
else:
    print(f"Phase 10.23 remains champion (MAE=43.64)")
print("="*90)
