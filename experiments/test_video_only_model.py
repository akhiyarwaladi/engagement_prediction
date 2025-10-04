#!/usr/bin/env python3
"""
Video-Only Model Analysis
==========================

Question: Do videos have different engagement patterns than photos?

Current situation:
- 53 videos (15.2% of dataset)
- Video features: duration, fps, frames, motion, brightness
- Previously: video features had -0.4% MAE impact when tested with all posts

Let's build a SEPARATE model just for videos to find:
1. Do video features predict VIDEO engagement better than text alone?
2. Which video features matter most (duration? motion? brightness?)
3. What's the optimal video duration for @fst_unja?
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "=" * 90)
print(" " * 25 + "VIDEO-ONLY MODEL ANALYSIS")
print(" " * 15 + "Do Videos Have Different Engagement Patterns?")
print("=" * 90)

# Load data
print("\n[DATA] Loading datasets...")
baseline_df = pd.read_csv('data/processed/baseline_dataset.csv')
bert_df = pd.read_csv('data/processed/bert_embeddings.csv')
enhanced_df = pd.read_csv('data/processed/enhanced_visual_features.csv')

bert_cols = [col for col in bert_df.columns if col.startswith('bert_dim_')]
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                'hour', 'day_of_week', 'is_weekend', 'month']  # Remove is_video
video_features = ['video_duration', 'video_fps', 'video_frames', 'video_motion', 'video_brightness']

# Filter VIDEOS ONLY
videos_df = baseline_df[baseline_df['is_video'] == 1].copy()
print(f"   Total posts: {len(baseline_df)}")
print(f"   Videos: {len(videos_df)} ({len(videos_df)/len(baseline_df)*100:.1f}%)")
print(f"   Photos: {len(baseline_df) - len(videos_df)}")

if len(videos_df) < 20:
    print("\n[WARNING] Too few videos ({len(videos_df)}) for reliable model training!")
    print("   Need at least 30-50 samples. Results may be unreliable.")

# Video statistics
y_videos = videos_df['likes'].copy()
print(f"\n[STATS] Video Engagement Statistics:")
print(f"   Mean likes: {y_videos.mean():.1f}")
print(f"   Median likes: {y_videos.median():.1f}")
print(f"   Std: {y_videos.std():.1f}")
print(f"   Min: {y_videos.min()}")
print(f"   Max: {y_videos.max()}")

# Photo statistics for comparison
photos_df = baseline_df[baseline_df['is_video'] == 0].copy()
y_photos = photos_df['likes'].copy()
print(f"\n[COMPARISON] Photo vs Video Engagement:")
print(f"   Photos avg: {y_photos.mean():.1f} likes")
print(f"   Videos avg: {y_videos.mean():.1f} likes")

difference = ((y_videos.mean() - y_photos.mean()) / y_photos.mean()) * 100
if difference > 0:
    print(f"   Videos get +{difference:.1f}% more likes on average!")
else:
    print(f"   Videos get {difference:.1f}% fewer likes on average")

# Get video features
enhanced_videos = enhanced_df[enhanced_df['is_video'] == 1].copy()

print(f"\n[VIDEO FEATURES] Video Feature Statistics:")
for vf in video_features:
    values = enhanced_videos[vf]
    print(f"   {vf:20} mean={values.mean():.2f}, median={values.median():.2f}, "
          f"min={values.min():.2f}, max={values.max():.2f}")

# Correlation analysis
print(f"\n[CORRELATION] Video Features vs Likes:")
video_data = videos_df[['post_id', 'likes']].merge(
    enhanced_videos[['post_id'] + video_features],
    on='post_id'
)

for vf in video_features:
    corr = video_data[[vf, 'likes']].corr().iloc[0, 1]
    print(f"   {vf:20} correlation={corr:+.4f}")

# BERT PCA for videos
print(f"\n[BERT] Extracting BERT features for videos...")
video_indices = videos_df.index
X_bert_videos = bert_df.loc[video_indices, bert_cols].copy()

pca_bert = PCA(n_components=min(30, len(videos_df)-1), random_state=42)  # Reduce components for small dataset
X_bert_reduced = pd.DataFrame(
    pca_bert.fit_transform(X_bert_videos),
    columns=[f'bert_pc_{i}' for i in range(min(30, len(videos_df)-1))]
)
print(f"   BERT: 768 -> {X_bert_reduced.shape[1]} dims ({pca_bert.explained_variance_ratio_.sum()*100:.1f}% variance)")

# Build models
experiments = []

# Config 1: Text-only (baseline for videos)
X_text_videos = pd.concat([
    videos_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced
], axis=1)
experiments.append({
    'name': 'Text-Only (Videos)',
    'features': X_text_videos,
    'video_features': []
})

# Config 2: Text + All Video Features
X_text_video_features = pd.concat([
    videos_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    enhanced_videos[video_features].reset_index(drop=True)
], axis=1)
experiments.append({
    'name': 'Text + All Video Features',
    'features': X_text_video_features,
    'video_features': video_features
})

# Config 3-7: Individual video features
for vf in video_features:
    X_temp = pd.concat([
        videos_df[baseline_cols].reset_index(drop=True),
        X_bert_reduced,
        enhanced_videos[[vf]].reset_index(drop=True)
    ], axis=1)
    experiments.append({
        'name': f'Text + {vf.replace("video_", "").title()}',
        'features': X_temp,
        'video_features': [vf]
    })

print(f"\n[CONFIG] Testing {len(experiments)} configurations on {len(videos_df)} videos:")
for i, exp in enumerate(experiments, 1):
    print(f"   {i}. {exp['name']} ({exp['features'].shape[1]} features)")

# Run experiments
print("\n" + "=" * 90)
print("RUNNING EXPERIMENTS (Small Dataset Warning: Results may vary!)")
print("=" * 90)

results = []

for exp in experiments:
    print(f"\n[TEST] {exp['name']}")
    X = exp['features']

    # Check if we have enough samples
    if len(X) < 20:
        print("   [SKIP] Too few samples for train/test split")
        continue

    # Train-test split (smaller test size for small dataset)
    test_size = max(0.2, 10 / len(X))  # At least 10 samples for test
    X_train_raw, X_test_raw, y_train_orig, y_test_orig = train_test_split(
        X, y_videos, test_size=test_size, random_state=42
    )

    print(f"   Train: {len(X_train_raw)}, Test: {len(X_test_raw)}")

    # Preprocessing
    clip_value = np.percentile(y_train_orig, 95)  # Lower percentile for small dataset
    y_train_clipped = np.clip(y_train_orig, None, clip_value)
    y_train_log = np.log1p(y_train_clipped)
    y_test_log = np.log1p(y_test_orig)

    transformer = QuantileTransformer(
        n_quantiles=min(10, len(X_train_raw)),  # Fewer quantiles
        output_distribution='normal',
        random_state=42
    )

    X_train = pd.DataFrame(
        transformer.fit_transform(X_train_raw),
        columns=X_train_raw.columns
    )
    X_test = pd.DataFrame(
        transformer.transform(X_test_raw),
        columns=X_test_raw.columns
    )

    # Train model (simpler model for small dataset)
    print("   Training...", end=" ")

    rf = RandomForestRegressor(
        n_estimators=100,  # Fewer trees
        max_depth=8,  # Shallower
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train_log)

    # Predictions
    pred_log = rf.predict(X_test)
    pred = np.expm1(pred_log)

    # Evaluate
    mae = mean_absolute_error(y_test_orig, pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, pred))
    r2 = r2_score(y_test_orig, pred)

    # Feature importance for video features
    feature_importance = pd.DataFrame({
        'feature': list(X.columns),
        'importance': rf.feature_importances_
    })

    video_importance = 0.0
    for vf in exp['video_features']:
        vf_imp = feature_importance[feature_importance['feature'] == vf]['importance'].values
        if len(vf_imp) > 0:
            video_importance += vf_imp[0]

    results.append({
        'name': exp['name'],
        'features': X.shape[1],
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'video_importance': video_importance
    })

    print(f"MAE={mae:.2f}, R2={r2:.4f}, Video={video_importance*100:.2f}%")

# Results summary
print("\n" + "=" * 90)
print("RESULTS SUMMARY - VIDEO-ONLY MODELS")
print("=" * 90)

baseline_mae = results[0]['mae']
baseline_r2 = results[0]['r2']

print("\n[TABLE] Performance Comparison:")
print("-" * 100)
print(f"{'Configuration':<35} | {'Features':>8} | {'MAE':>8} | {'R2':>8} | {'vs Text MAE':>12} | {'Video %':>9}")
print("-" * 100)

for res in results:
    mae_change = ((baseline_mae - res['mae']) / baseline_mae) * 100
    symbol = "+" if mae_change > 0 else ""

    print(f"{res['name']:<35} | {res['features']:>8} | {res['mae']:>8.2f} | "
          f"{res['r2']:>8.4f} | {symbol}{mae_change:>11.2f}% | "
          f"{res['video_importance']*100:>8.2f}%")

print("-" * 100)

# Analysis
print("\n" + "=" * 90)
print("ANALYSIS")
print("=" * 90)

best_model = min(results[1:], key=lambda x: x['mae'])  # Best non-baseline
print(f"\n[BEST] Best video model: {best_model['name']}")
print(f"   MAE: {best_model['mae']:.2f} ({((baseline_mae-best_model['mae'])/baseline_mae)*100:+.2f}% vs text-only)")
print(f"   R2: {best_model['r2']:.4f}")

if best_model['mae'] < baseline_mae:
    print(f"\n[VERDICT] Video features HELP predict video engagement!")
    print(f"   Improvement: {((baseline_mae-best_model['mae'])/baseline_mae)*100:.1f}%")
else:
    print(f"\n[VERDICT] Video features DON'T help (text-only is better)")
    print(f"   Degradation: {((best_model['mae']-baseline_mae)/baseline_mae)*100:.1f}%")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('experiments/video_only_model_results.csv', index=False)
print(f"\n[SAVE] Results saved to: experiments/video_only_model_results.csv")

print("\n" + "=" * 90)
print("VIDEO-ONLY ANALYSIS COMPLETE!")
print("=" * 90)
print(f"\n[NOTE] Small sample size ({len(videos_df)} videos) - results may not be statistically significant")
print("")
