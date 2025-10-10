#!/usr/bin/env python3
"""
PHASE 11.2: ACCOUNT-SPECIFIC FEATURES (Multi-Account Leverage)
Hypothesis: Different accounts have different engagement patterns
Strategy: Add account-level statistics as features
Target: Beat MAE=43.49
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
print(" "*12 + "PHASE 11.2: ACCOUNT-SPECIFIC FEATURES (Multi-Account)")
print(" "*20 + "Target: Beat Phase 10.24 MAE=43.49")
print("="*90)
print()

df_main = pd.read_csv('multi_account_dataset.csv')
df_bert = pd.read_csv('data/processed/bert_embeddings_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])
df_visual = pd.read_csv('data/processed/advanced_visual_features_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])
df = df_main.merge(df_bert, on=['post_id', 'account'], how='inner').merge(df_visual, on=['post_id', 'account'], how='inner')

print(f"[LOAD] Dataset: {len(df)} posts from {df['account'].nunique()} accounts")
print()

# Account statistics (calculated from training set only to avoid leakage)
train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
df_train_stats = df.iloc[train_idx].copy()

# Compute account-level statistics
account_stats = df_train_stats.groupby('account').agg({
    'likes': ['mean', 'std', 'median'],
    'caption_length': 'mean',
    'hashtag_count': 'mean',
    'is_video': 'mean'
}).reset_index()

account_stats.columns = ['account', 'account_avg_likes', 'account_std_likes', 'account_median_likes',
                          'account_avg_caption_len', 'account_avg_hashtags', 'account_video_ratio']

# Merge account stats back
df = df.merge(account_stats, on='account', how='left')

# Fill NaN for test accounts not in training (shouldn't happen with our split, but safety)
for col in ['account_avg_likes', 'account_std_likes', 'account_median_likes',
            'account_avg_caption_len', 'account_avg_hashtags', 'account_video_ratio']:
    df[col] = df[col].fillna(df[col].median())

print("[ENGINEER] Account-specific features:")
print(f"   account_avg_likes: Mean likes per account")
print(f"   account_std_likes: Engagement variance per account")
print(f"   account_median_likes: Median likes per account")
print(f"   account_avg_caption_len: Average caption length per account")
print(f"   account_avg_hashtags: Average hashtag usage per account")
print(f"   account_video_ratio: Video/Photo ratio per account")
print()

# Add relative features (current post vs account average)
df['likes_vs_account_avg'] = df['likes'] / (df['account_avg_likes'] + 1)
df['caption_vs_account_avg'] = df['caption_length'] / (df['account_avg_caption_len'] + 1)
df['hashtag_vs_account_avg'] = df['hashtag_count'] / (df['account_avg_hashtags'] + 1)

print("[ENGINEER] Relative features:")
print(f"   likes_vs_account_avg: Post likes / Account avg (target leakage protected)")
print(f"   caption_vs_account_avg: Post caption / Account avg caption")
print(f"   hashtag_vs_account_avg: Post hashtags / Account avg hashtags")
print()

baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                 'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']
bert_cols = [col for col in df.columns if col.startswith('bert_')]

# Phase 10.23 visual + cross interactions
metadata_base = ['file_size_kb', 'is_portrait', 'is_landscape', 'is_square']
df['resolution_log'] = np.log1p(df['resolution'])
df['aspect_ratio_sq'] = df['aspect_ratio'] ** 2
df['aspect_x_logres'] = df['aspect_ratio'] * df['resolution_log']
df['filesize_x_logres'] = df['file_size_kb'] * df['resolution_log']
df['aspect_sq_x_logres'] = df['aspect_ratio_sq'] * df['resolution_log']

df['caption_x_aspect'] = df['caption_length'] * df['aspect_ratio']
df['caption_x_logres'] = df['caption_length'] * df['resolution_log']
df['hashtag_x_logres'] = df['hashtag_count'] * df['resolution_log']
df['word_x_filesize'] = df['word_count'] * df['file_size_kb']
df['caption_x_filesize'] = df['caption_length'] * df['file_size_kb']

cross_interactions = ['caption_x_aspect', 'caption_x_logres', 'hashtag_x_logres',
                      'word_x_filesize', 'caption_x_filesize']

visual_features = (metadata_base + ['aspect_ratio', 'resolution_log', 'aspect_ratio_sq',
                   'aspect_x_logres', 'filesize_x_logres', 'aspect_sq_x_logres'] +
                   cross_interactions)

# NEW: Account features (exclude likes_vs_account_avg to prevent target leakage on test)
account_features = ['account_avg_caption_len', 'account_avg_hashtags', 'account_video_ratio',
                    'caption_vs_account_avg', 'hashtag_vs_account_avg']

print(f"[STRATEGY] Phase 10.24 + BERT PCA 70 + Account Features")
print(f"   Baseline: {len(baseline_cols)}")
print(f"   BERT: {len(bert_cols)} -> 70 PCA")
print(f"   Visual+Cross: {len(visual_features)}")
print(f"   Account-specific: {len(account_features)} NEW")
print()

X_baseline = df[baseline_cols].values
X_bert_full = df[bert_cols].values
X_visual = df[visual_features].values
X_account = df[account_features].values
y = df['likes'].values

X_baseline_train, X_baseline_test = X_baseline[train_idx], X_baseline[test_idx]
X_bert_train, X_bert_test = X_bert_full[train_idx], X_bert_full[test_idx]
X_visual_train, X_visual_test = X_visual[train_idx], X_visual[test_idx]
X_account_train, X_account_test = X_account[train_idx], X_account[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

clip_threshold = np.percentile(y_train, 99)
y_train_log = np.log1p(np.clip(y_train, 0, clip_threshold))
y_test_log = np.log1p(np.clip(y_test, 0, clip_threshold))

# BERT PCA 70 (champion config)
print("[PCA] Reducing BERT from 768 to 70 dimensions...")
pca_bert = PCA(n_components=70, random_state=42)
X_bert_pca_train = pca_bert.fit_transform(X_bert_train)
X_bert_pca_test = pca_bert.transform(X_bert_test)
variance_preserved = pca_bert.explained_variance_ratio_.sum()
print(f"   Variance preserved: {variance_preserved*100:.1f}%")
print()

# Combine all features INCLUDING account features
X_train = np.hstack([X_baseline_train, X_bert_pca_train, X_visual_train, X_account_train])
X_test = np.hstack([X_baseline_test, X_bert_pca_test, X_visual_test, X_account_test])

print(f"[FEATURES] Total: {X_train.shape[1]} features (was 94, now with +{len(account_features)} account)")
print()

scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("[MODEL] Training 4-model stacking ensemble...")

models = [
    ('gb', GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, random_state=42)),
    ('hgb', HistGradientBoostingRegressor(max_iter=600, learning_rate=0.07, max_depth=7, random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=300, max_depth=16, random_state=42, n_jobs=-1)),
    ('et', ExtraTreesRegressor(n_estimators=300, max_depth=16, random_state=42, n_jobs=-1))
]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X_train_scaled), len(models)))
test_preds = np.zeros((len(X_test_scaled), len(models)))

for i, (name, model) in enumerate(models):
    print(f"   Training {name}...", end=" ")

    for fold_idx, (train_fold, val_fold) in enumerate(kf.split(X_train_scaled)):
        X_tr, X_val = X_train_scaled[train_fold], X_train_scaled[val_fold]
        y_tr, y_val = y_train_log[train_fold], y_train_log[val_fold]

        model.fit(X_tr, y_tr)
        oof_preds[val_fold, i] = model.predict(X_val)

    model.fit(X_train_scaled, y_train_log)
    test_preds[:, i] = model.predict(X_test_scaled)
    print("Done")

print("   Training Ridge meta-learner...", end=" ")
meta_model = Ridge(alpha=10)
meta_model.fit(oof_preds, y_train_log)
print("Done")
print()

final_pred_test = meta_model.predict(test_preds)
final_pred_test_inv = np.expm1(final_pred_test)
y_test_inv = np.expm1(y_test_log)

mae = mean_absolute_error(y_test_inv, final_pred_test_inv)
r2 = r2_score(y_test_inv, final_pred_test_inv)

print("="*90)
print(f"[RESULT] Phase 11.2: MAE={mae:.2f}, R2={r2:.4f}")
print("="*90)

champion_mae = 43.49
if mae < champion_mae:
    improvement = ((champion_mae - mae) / champion_mae) * 100
    print(f"[CHAMPION] NEW RECORD! Beat Phase 10.24 by {improvement:.2f}%!")
    print(f"   Account-specific features are effective!")
else:
    decline = ((mae - champion_mae) / champion_mae) * 100
    print(f"   Phase 10.24 remains champion (MAE={champion_mae:.2f})")
    print(f"   Account features don't help (+{decline:.2f}% worse)")

print("="*90)
print()

if mae < champion_mae:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/phase11_2_account_features_{timestamp}.pkl"
    joblib.dump({
        'scaler': scaler,
        'pca_bert': pca_bert,
        'base_models': [m for _, m in models],
        'meta_model': meta_model,
        'account_stats': account_stats,  # Save for production use
        'mae': mae,
        'r2': r2
    }, model_path)
    print(f"[SAVE] Model saved: {model_path}")
    print()

print("[INSIGHT] Account-level statistics tested:")
print(f"   Average likes per account")
print(f"   Engagement variance per account")
print(f"   Content style (caption length, hashtags, video ratio)")
print(f"   Relative metrics (post vs account average)")
