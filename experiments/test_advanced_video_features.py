#!/usr/bin/env python3
"""
Test Advanced Video Features
Compare temporal video analysis with previous approaches
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "="*90)
print(" "*25 + "ADVANCED VIDEO FEATURES TEST")
print(" "*15 + "Temporal, Motion, Scene Analysis vs Previous Best")
print("="*90)

# Load data
baseline_df = pd.read_csv('data/processed/baseline_dataset.csv')
bert_df = pd.read_csv('data/processed/bert_embeddings.csv')
aesthetic_df = pd.read_csv('data/processed/aesthetic_features.csv')
video_df = pd.read_csv('data/processed/advanced_video_features.csv')

merged_df = baseline_df.merge(aesthetic_df, on='post_id', how='left')
merged_df = merged_df.merge(video_df, on='post_id', how='left')

# BERT PCA
bert_cols = [col for col in bert_df.columns if col.startswith('bert_dim_')]
pca_bert = PCA(n_components=50, random_state=42)
X_bert_reduced = pd.DataFrame(
    pca_bert.fit_transform(bert_df[bert_cols]),
    columns=[f'bert_pc_{i}' for i in range(50)]
)

baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']

nima_features = [
    'aesthetic_sharpness', 'aesthetic_noise', 'aesthetic_brightness',
    'aesthetic_exposure_quality', 'aesthetic_color_harmony',
    'aesthetic_saturation', 'aesthetic_saturation_variance',
    'aesthetic_luminance_contrast'
]

# Advanced video features (21)
video_features = [col for col in video_df.columns if col.startswith('video_')]

# Fill NaN
for col in nima_features:
    merged_df[col] = merged_df[col].fillna(0)
for col in video_features:
    merged_df[col] = merged_df[col].fillna(0)

y = merged_df['likes'].copy()

print(f"\n[DATA] Total posts: {len(merged_df)}")
print(f"   Videos: {(merged_df['is_video'] == 1).sum()}")
print(f"   Advanced video features: {len(video_features)}")

# Experiments
experiments = []

# 1. Champion: Text + NIMA
X_champion = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    merged_df[nima_features].reset_index(drop=True)
], axis=1)
experiments.append({'name': 'Champion (Text + NIMA)', 'features': X_champion})

# 2. Text + Advanced Video Features
X_text_video = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    merged_df[video_features].reset_index(drop=True)
], axis=1)
experiments.append({'name': 'Text + Advanced Video (21)', 'features': X_text_video})

# 3. Text + NIMA + Advanced Video
X_nima_video = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    merged_df[nima_features].reset_index(drop=True),
    merged_df[video_features].reset_index(drop=True)
], axis=1)
experiments.append({'name': 'NIMA + Advanced Video (29)', 'features': X_nima_video})

# 4. Top video features (motion, optical flow, scene changes)
top_video = [
    'video_motion_mean', 'video_motion_std', 'video_motion_max',
    'video_optical_flow_mean', 'video_optical_flow_max',
    'video_scene_change_rate', 'video_edge_complexity_mean'
]
X_top_video = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    merged_df[nima_features].reset_index(drop=True),
    merged_df[top_video].reset_index(drop=True)
], axis=1)
experiments.append({'name': 'NIMA + Top Video (7)', 'features': X_top_video})

# 5. Video temporal features only (motion + scene)
temporal_video = [
    'video_motion_mean', 'video_motion_std', 'video_motion_pacing',
    'video_scene_change_rate', 'video_optical_flow_mean'
]
X_temporal = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    merged_df[nima_features].reset_index(drop=True),
    merged_df[temporal_video].reset_index(drop=True)
], axis=1)
experiments.append({'name': 'NIMA + Temporal Video (5)', 'features': X_temporal})

# 6. Add interactions (manual + non-linear) - previous best
manual_interactions = pd.DataFrame({
    'caption_x_hashtag': baseline_df['caption_length'] * baseline_df['hashtag_count'],
    'caption_x_sharp': baseline_df['caption_length'] * merged_df['aesthetic_sharpness'].fillna(0),
    'video_x_sharp': baseline_df['is_video'] * merged_df['aesthetic_sharpness'].fillna(0),
    'hour_x_weekend': baseline_df['hour'] * baseline_df['is_weekend'],
    'sharp_x_contrast': merged_df['aesthetic_sharpness'].fillna(0) * merged_df['aesthetic_luminance_contrast'].fillna(0),
    'sharp_x_saturation': merged_df['aesthetic_sharpness'].fillna(0) * merged_df['aesthetic_saturation'].fillna(0),
    'saturation_x_brightness': merged_df['aesthetic_saturation'].fillna(0) * merged_df['aesthetic_brightness'].fillna(0),
})

nonlinear_features = pd.DataFrame({
    'sharp_squared': merged_df['aesthetic_sharpness'].fillna(0) ** 2,
    'sharp_sqrt': np.sqrt(merged_df['aesthetic_sharpness'].fillna(0)),
    'saturation_squared': merged_df['aesthetic_saturation'].fillna(0) ** 2,
    'caption_squared': baseline_df['caption_length'] ** 2,
    'caption_log': np.log1p(baseline_df['caption_length']),
})

X_ultimate = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    merged_df[nima_features].reset_index(drop=True),
    merged_df[top_video].reset_index(drop=True),
    manual_interactions.reset_index(drop=True),
    nonlinear_features.reset_index(drop=True)
], axis=1)
experiments.append({'name': 'ULTIMATE: NIMA + Video + Interactions', 'features': X_ultimate})

print(f"\n[CONFIG] Testing {len(experiments)} configurations")

# Run experiments
results = []

for exp in experiments:
    X = exp['features']
    X_train_raw, X_test_raw, y_train_orig, y_test_orig = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clip_value = np.percentile(y_train_orig, 99)
    y_train_clipped = np.clip(y_train_orig, None, clip_value)
    y_train_log = np.log1p(y_train_clipped)

    transformer = QuantileTransformer(n_quantiles=min(100, len(X_train_raw)),
                                     output_distribution='normal', random_state=42)
    X_train = pd.DataFrame(transformer.fit_transform(X_train_raw), columns=X_train_raw.columns)
    X_test = pd.DataFrame(transformer.transform(X_test_raw), columns=X_test_raw.columns)

    rf = RandomForestRegressor(n_estimators=250, max_depth=14, min_samples_split=3,
                              min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_log)

    hgb = HistGradientBoostingRegressor(max_iter=400, max_depth=14, learning_rate=0.05,
                                       min_samples_leaf=4, l2_regularization=0.1, random_state=42,
                                       early_stopping=True, validation_fraction=0.2, n_iter_no_change=20)
    hgb.fit(X_train, y_train_log)

    pred_log = 0.5 * rf.predict(X_test) + 0.5 * hgb.predict(X_test)
    pred = np.expm1(pred_log)

    mae = mean_absolute_error(y_test_orig, pred)
    r2 = r2_score(y_test_orig, pred)

    results.append({'name': exp['name'], 'features': X.shape[1], 'mae': mae, 'r2': r2})

# Results
champion_mae = results[0]['mae']

print("\n" + "="*90)
print("RESULTS - ADVANCED VIDEO FEATURES")
print("="*90)
print(f"\n{'Configuration':<45} | {'Feat':>5} | {'MAE':>8} | {'vs Champ':>10} | {'R2':>8}")
print("-" * 95)

for res in results:
    change = ((champion_mae - res['mae']) / champion_mae) * 100
    symbol = "+" if change > 0 else ""
    print(f"{res['name']:<45} | {res['features']:>5} | {res['mae']:>8.2f} | "
          f"{symbol}{change:>9.2f}% | {res['r2']:>8.4f}")

print("-" * 95)

# Find best
best = min(results, key=lambda x: x['mae'])
print(f"\n[BEST] {best['name']}")
print(f"   MAE: {best['mae']:.2f}")
print(f"   R2: {best['r2']:.4f}")
print(f"   Features: {best['features']}")
improvement = ((champion_mae - best['mae']) / champion_mae) * 100
print(f"   Improvement vs champion: {improvement:+.2f}%")

if improvement > 0:
    print(f"\n   [SUCCESS] Advanced video features improved by {improvement:.2f}%!")
else:
    print(f"\n   [INFO] Champion still best.")

# Save
results_df = pd.DataFrame(results)
results_df.to_csv('experiments/advanced_video_features_results.csv', index=False)
print(f"\n[SAVE] Results saved to: experiments/advanced_video_features_results.csv")

print("\n" + "="*90)
print("ADVANCED VIDEO FEATURES TEST COMPLETE!")
print("="*90)
print("")
