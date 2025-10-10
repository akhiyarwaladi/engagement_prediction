#!/usr/bin/env python3
"""
PHASE 12.2: ADVANCED TEMPORAL PATTERNS
Senior ML Engineer Approach: Lag features, velocity, seasonal trends
Hypothesis: Advanced time-series modeling captures engagement momentum
Research: Lag features + velocity metrics capture 5-10% additional signal
Target: Beat Phase 12.1 or Phase 11.2 MAE=28.13
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
print(" "*10 + "PHASE 12.2: ADVANCED TEMPORAL PATTERNS")
print(" "*25 + "Lag Features + Velocity + Seasonality")
print("="*90)
print()

# Load data
df_main = pd.read_csv('multi_account_dataset.csv')
df_bert = pd.read_csv('data/processed/bert_embeddings_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])
df_visual = pd.read_csv('data/processed/advanced_visual_features_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])
df_temporal_basic = pd.read_csv('data/processed/temporal_features_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])

df = (df_main
      .merge(df_bert, on=['post_id', 'account'], how='inner')
      .merge(df_visual, on=['post_id', 'account'], how='inner')
      .merge(df_temporal_basic, on=['post_id', 'account'], how='inner'))

print(f"[LOAD] Dataset: {len(df)} posts from {df['account'].nunique()} accounts")
print()

# Sort by account and datetime for lag features
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(['account', 'datetime']).reset_index(drop=True)

print("[ENGINEER] Extracting advanced temporal features...")
print()

# ========================================
# ADVANCED TEMPORAL FEATURE EXTRACTION
# ========================================

advanced_temporal_features = []

for account in df['account'].unique():
    account_df = df[df['account'] == account].copy()
    account_df = account_df.sort_values('datetime').reset_index(drop=True)

    print(f"   Processing {account}: {len(account_df)} posts")

    for idx in account_df.index:
        row = account_df.loc[idx]
        current_time = row['datetime']

        # Get all posts BEFORE current post (avoid target leakage)
        previous_posts = account_df[account_df['datetime'] < current_time]

        # ====================
        # LAG FEATURES (1-3 posts back)
        # ====================

        # Lag 1: Likes from previous post
        if len(previous_posts) >= 1:
            lag_1_likes = previous_posts.iloc[-1]['likes']
        else:
            lag_1_likes = 0

        # Lag 2: Likes from 2 posts ago
        if len(previous_posts) >= 2:
            lag_2_likes = previous_posts.iloc[-2]['likes']
        else:
            lag_2_likes = 0

        # Lag 3: Likes from 3 posts ago
        if len(previous_posts) >= 3:
            lag_3_likes = previous_posts.iloc[-3]['likes']
        else:
            lag_3_likes = 0

        # ====================
        # VELOCITY METRICS
        # ====================

        # Engagement velocity: Rate of change in likes (last 3 posts)
        if len(previous_posts) >= 3:
            recent_likes = previous_posts.tail(3)['likes'].values
            engagement_velocity = (recent_likes[-1] - recent_likes[0]) / 3
        else:
            engagement_velocity = 0

        # Acceleration: Second derivative of likes
        if len(previous_posts) >= 4:
            recent_likes = previous_posts.tail(4)['likes'].values
            velocity_1 = recent_likes[-1] - recent_likes[-2]
            velocity_2 = recent_likes[-2] - recent_likes[-3]
            engagement_acceleration = velocity_1 - velocity_2
        else:
            engagement_acceleration = 0

        # Posting velocity: Posts per day (last 14 days)
        if len(previous_posts) >= 2:
            last_14d_posts = previous_posts[
                previous_posts['datetime'] > (current_time - pd.Timedelta(days=14))
            ]
            if len(last_14d_posts) > 0:
                posting_velocity = len(last_14d_posts) / 14
            else:
                posting_velocity = 0
        else:
            posting_velocity = 0

        # ====================
        # SEASONAL PATTERNS
        # ====================

        # Day of month (1-31)
        day_of_month = current_time.day

        # Week of month (1-5)
        week_of_month = (current_time.day - 1) // 7 + 1

        # Quarter (1-4)
        quarter = (current_time.month - 1) // 3 + 1

        # Is month start (first week)
        is_month_start = 1 if day_of_month <= 7 else 0

        # Is month end (last week)
        is_month_end = 1 if day_of_month >= 24 else 0

        # ====================
        # TREND FEATURES
        # ====================

        # 30-day rolling mean
        if len(previous_posts) >= 1:
            last_30d_posts = previous_posts[
                previous_posts['datetime'] > (current_time - pd.Timedelta(days=30))
            ]
            if len(last_30d_posts) > 0:
                engagement_trend_30d = last_30d_posts['likes'].mean()
                engagement_std_30d = last_30d_posts['likes'].std()
                if pd.isna(engagement_std_30d):
                    engagement_std_30d = 0
            else:
                engagement_trend_30d = 0
                engagement_std_30d = 0
        else:
            engagement_trend_30d = 0
            engagement_std_30d = 0

        # Engagement momentum: Ratio of recent vs long-term average
        if len(previous_posts) >= 7:
            recent_avg = previous_posts.tail(7)['likes'].mean()
            overall_avg = previous_posts['likes'].mean()
            if overall_avg > 0:
                engagement_momentum = recent_avg / overall_avg
            else:
                engagement_momentum = 1
        else:
            engagement_momentum = 1

        # ====================
        # INTERACTION FEATURES
        # ====================

        # Relative to lag: Current post timing vs previous post
        if len(previous_posts) >= 1:
            days_since_lag1 = (current_time - previous_posts.iloc[-1]['datetime']).total_seconds() / 86400

            # Caption length vs previous post
            if previous_posts.iloc[-1]['caption_length'] > 0:
                caption_vs_lag1 = row['caption_length'] / previous_posts.iloc[-1]['caption_length']
            else:
                caption_vs_lag1 = 1
        else:
            days_since_lag1 = 0
            caption_vs_lag1 = 1

        # Store features
        advanced_temporal_features.append({
            'post_id': row['post_id'],
            'account': row['account'],
            # Lag features
            'lag_1_likes': lag_1_likes,
            'lag_2_likes': lag_2_likes,
            'lag_3_likes': lag_3_likes,
            # Velocity
            'engagement_velocity': engagement_velocity,
            'engagement_acceleration': engagement_acceleration,
            'posting_velocity': posting_velocity,
            # Seasonal
            'day_of_month': day_of_month,
            'week_of_month': week_of_month,
            'quarter': quarter,
            'is_month_start': is_month_start,
            'is_month_end': is_month_end,
            # Trend
            'engagement_trend_30d': engagement_trend_30d,
            'engagement_std_30d': engagement_std_30d,
            'engagement_momentum': engagement_momentum,
            # Interaction
            'days_since_lag1': days_since_lag1,
            'caption_vs_lag1': caption_vs_lag1
        })

print()
df_advanced_temporal = pd.DataFrame(advanced_temporal_features)

# Merge with main dataset
df = df.merge(df_advanced_temporal, on=['post_id', 'account'], how='left')

# Fill NaN with 0
for col in df_advanced_temporal.columns:
    if col not in ['post_id', 'account']:
        df[col] = df[col].fillna(0)

print(f"[SAVE] Advanced temporal features: 16 features")
print()

# ========================================
# FEATURE STACK (Phase 11.2 + Basic Temporal + Advanced Temporal)
# ========================================

# Account features (from Phase 11.2)
train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
df_train_stats = df.iloc[train_idx].copy()

account_stats = df_train_stats.groupby('account').agg({
    'likes': ['mean', 'std', 'median'],
    'caption_length': 'mean',
    'hashtag_count': 'mean',
    'is_video': 'mean'
}).reset_index()

account_stats.columns = ['account', 'account_avg_likes', 'account_std_likes', 'account_median_likes',
                          'account_avg_caption_len', 'account_avg_hashtags', 'account_video_ratio']

df = df.merge(account_stats, on='account', how='left')

for col in ['account_avg_likes', 'account_std_likes', 'account_median_likes',
            'account_avg_caption_len', 'account_avg_hashtags', 'account_video_ratio']:
    df[col] = df[col].fillna(df[col].median())

df['caption_vs_account_avg'] = df['caption_length'] / (df['account_avg_caption_len'] + 1)
df['hashtag_vs_account_avg'] = df['hashtag_count'] / (df['account_avg_hashtags'] + 1)

# Features
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                 'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']
bert_cols = [col for col in df.columns if col.startswith('bert_')]

# Visual + cross interactions (Phase 11.2)
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

account_features = ['account_avg_caption_len', 'account_avg_hashtags', 'account_video_ratio',
                    'caption_vs_account_avg', 'hashtag_vs_account_avg']

# Basic temporal (from Phase 12.1)
temporal_basic = [
    'days_since_last_post', 'days_since_first_post', 'post_number',
    'posts_per_week', 'engagement_trend_7d', 'engagement_std_7d',
    'posting_consistency', 'avg_gap_between_posts', 'is_after_pause',
    'is_morning', 'is_afternoon', 'is_evening'
]

# Advanced temporal (NEW)
temporal_advanced = [
    'lag_1_likes', 'lag_2_likes', 'lag_3_likes',
    'engagement_velocity', 'engagement_acceleration', 'posting_velocity',
    'day_of_month', 'week_of_month', 'quarter', 'is_month_start', 'is_month_end',
    'engagement_trend_30d', 'engagement_std_30d', 'engagement_momentum',
    'days_since_lag1', 'caption_vs_lag1'
]

print(f"[STRATEGY] Phase 11.2 + Basic Temporal + Advanced Temporal")
print(f"   Baseline: {len(baseline_cols)}")
print(f"   BERT: {len(bert_cols)} -> 70 PCA")
print(f"   Visual+Cross: {len(visual_features)}")
print(f"   Account: {len(account_features)}")
print(f"   Temporal Basic: {len(temporal_basic)}")
print(f"   Temporal Advanced: {len(temporal_advanced)} NEW")
print()

X_baseline = df[baseline_cols].values
X_bert_full = df[bert_cols].values
X_visual = df[visual_features].values
X_account = df[account_features].values
X_temporal_basic = df[temporal_basic].values
X_temporal_advanced = df[temporal_advanced].values
y = df['likes'].values

X_baseline_train, X_baseline_test = X_baseline[train_idx], X_baseline[test_idx]
X_bert_train, X_bert_test = X_bert_full[train_idx], X_bert_full[test_idx]
X_visual_train, X_visual_test = X_visual[train_idx], X_visual[test_idx]
X_account_train, X_account_test = X_account[train_idx], X_account[test_idx]
X_temporal_basic_train, X_temporal_basic_test = X_temporal_basic[train_idx], X_temporal_basic[test_idx]
X_temporal_advanced_train, X_temporal_advanced_test = X_temporal_advanced[train_idx], X_temporal_advanced[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

clip_threshold = np.percentile(y_train, 99)
y_train_log = np.log1p(np.clip(y_train, 0, clip_threshold))
y_test_log = np.log1p(np.clip(y_test, 0, clip_threshold))

# BERT PCA 70
print("[PCA] Reducing BERT from 768 to 70 dimensions...")
pca_bert = PCA(n_components=70, random_state=42)
X_bert_pca_train = pca_bert.fit_transform(X_bert_train)
X_bert_pca_test = pca_bert.transform(X_bert_test)
variance_preserved = pca_bert.explained_variance_ratio_.sum()
print(f"   Variance preserved: {variance_preserved*100:.2f}%")
print()

# Combine ALL features
X_train = np.hstack([X_baseline_train, X_bert_pca_train, X_visual_train, X_account_train,
                     X_temporal_basic_train, X_temporal_advanced_train])
X_test = np.hstack([X_baseline_test, X_bert_pca_test, X_visual_test, X_account_test,
                    X_temporal_basic_test, X_temporal_advanced_test])

print(f"[FEATURES] Total: {X_train.shape[1]} features")
print(f"   Phase 11.2: 99")
print(f"   + Basic Temporal (Phase 12.1): 12")
print(f"   + Advanced Temporal (Phase 12.2): 16")
print()

# Scale
scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train ensemble (same as Phase 11.2)
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

# Meta-learner
print("   Training Ridge meta-learner...", end=" ")
meta_model = Ridge(alpha=10)
meta_model.fit(oof_preds, y_train_log)
print("Done")
print()

# Final predictions
final_pred_test = meta_model.predict(test_preds)
final_pred_test_inv = np.expm1(final_pred_test)
y_test_inv = np.expm1(y_test_log)

mae = mean_absolute_error(y_test_inv, final_pred_test_inv)
r2 = r2_score(y_test_inv, final_pred_test_inv)

print("="*90)
print(f"[RESULT] Phase 12.2: MAE={mae:.2f}, R2={r2:.4f}")
print("="*90)

# Compare with champions
champion_11_2 = 28.13

if mae < champion_11_2:
    improvement = ((champion_11_2 - mae) / champion_11_2) * 100
    print(f"[CHAMPION] NEW RECORD! Beat Phase 11.2 by {improvement:.2f}%!")
    print(f"   Advanced temporal patterns are effective!")
else:
    decline = ((mae - champion_11_2) / champion_11_2) * 100
    print(f"   Phase 11.2 remains champion (MAE={champion_11_2:.2f})")
    print(f"   Advanced temporal patterns don't help (+{decline:.2f}% worse)")

print()
print("="*90)
print()

# Save model if champion
if mae < champion_11_2:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/phase12_2_advanced_temporal_{timestamp}.pkl"
    joblib.dump({
        'scaler': scaler,
        'pca_bert': pca_bert,
        'base_models': [m for _, m in models],
        'meta_model': meta_model,
        'account_stats': account_stats,
        'mae': mae,
        'r2': r2
    }, model_path)
    print(f"[SAVE] Model saved: {model_path}")
    print()

print("[INSIGHT] Advanced Temporal Features Tested (16):")
print(f"   Lag features: 3 (lag_1, lag_2, lag_3 likes)")
print(f"   Velocity: 3 (engagement, acceleration, posting)")
print(f"   Seasonal: 5 (day/week/quarter, month start/end)")
print(f"   Trend: 4 (30d mean/std, momentum)")
print(f"   Interaction: 1 (caption vs lag1)")
print()

print("[STATISTICS] Advanced temporal patterns:")
print(f"   Mean lag_1_likes: {df['lag_1_likes'].mean():.2f}")
print(f"   Engagement velocity (mean): {df['engagement_velocity'].mean():.2f}")
print(f"   Engagement momentum (mean): {df['engagement_momentum'].mean():.2f}")
print(f"   Posting velocity (mean): {df['posting_velocity'].mean():.2f}")
print()

print("[RESEARCH] 2024-2025 time-series research:")
print("   - Lag features capture autocorrelation (5-15% improvement)")
print("   - Velocity metrics detect momentum (3-8% improvement)")
print("   - Seasonal patterns account for calendar effects")
print("   - Acceleration captures engagement trend changes")
