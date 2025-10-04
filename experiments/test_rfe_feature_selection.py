#!/usr/bin/env python3
"""
Recursive Feature Elimination (RFE) Test
Systematically select best features from ULTIMATE model
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "="*90)
print(" "*25 + "RFE FEATURE SELECTION TEST")
print(" "*15 + "Recursive Feature Elimination on ULTIMATE Model")
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

top_video = [
    'video_motion_mean', 'video_motion_std', 'video_motion_max',
    'video_optical_flow_mean', 'video_optical_flow_max',
    'video_scene_change_rate', 'video_edge_complexity_mean'
]

# Fill NaN
for col in nima_features:
    merged_df[col] = merged_df[col].fillna(0)
for col in top_video:
    merged_df[col] = merged_df[col].fillna(0)

# Manual interactions
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

# Build ULTIMATE feature set
X_ultimate = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    merged_df[nima_features].reset_index(drop=True),
    merged_df[top_video].reset_index(drop=True),
    manual_interactions.reset_index(drop=True),
    nonlinear_features.reset_index(drop=True)
], axis=1)

y = merged_df['likes'].copy()

print(f"\n[DATA] Total features: {X_ultimate.shape[1]}")
print(f"   Baseline: {len(baseline_cols)}")
print(f"   BERT PCA: 50")
print(f"   NIMA: {len(nima_features)}")
print(f"   Video: {len(top_video)}")
print(f"   Interactions: {manual_interactions.shape[1]}")
print(f"   Non-linear: {nonlinear_features.shape[1]}")

# Train-test split
X_train_raw, X_test_raw, y_train_orig, y_test_orig = train_test_split(
    X_ultimate, y, test_size=0.2, random_state=42
)

# Preprocessing
clip_value = np.percentile(y_train_orig, 99)
y_train_clipped = np.clip(y_train_orig, None, clip_value)
y_train_log = np.log1p(y_train_clipped)
y_test_log = np.log1p(y_test_orig)

transformer = QuantileTransformer(n_quantiles=min(100, len(X_train_raw)),
                                 output_distribution='normal', random_state=42)
X_train = pd.DataFrame(transformer.fit_transform(X_train_raw), columns=X_train_raw.columns)
X_test = pd.DataFrame(transformer.transform(X_test_raw), columns=X_test_raw.columns)

# Baseline: ALL features
rf_full = RandomForestRegressor(n_estimators=250, max_depth=14, min_samples_split=3,
                               min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1)
rf_full.fit(X_train, y_train_log)

hgb_full = HistGradientBoostingRegressor(max_iter=400, max_depth=14, learning_rate=0.05,
                                        min_samples_leaf=4, l2_regularization=0.1, random_state=42,
                                        early_stopping=True, validation_fraction=0.2, n_iter_no_change=20)
hgb_full.fit(X_train, y_train_log)

pred_log_full = 0.5 * rf_full.predict(X_test) + 0.5 * hgb_full.predict(X_test)
pred_full = np.expm1(pred_log_full)
mae_full = mean_absolute_error(y_test_orig, pred_full)
r2_full = r2_score(y_test_orig, pred_full)

print(f"\n[BASELINE] All features ({X_ultimate.shape[1]})")
print(f"   MAE: {mae_full:.2f}")
print(f"   R2: {r2_full:.4f}")

# Test different feature counts with RFE
print(f"\n[RFE] Testing different feature counts...")

feature_counts = [80, 75, 70, 65, 60, 55, 50]
rfe_results = []

for n_features in feature_counts:
    print(f"\n   Testing {n_features} features...", end=" ")

    # RFE with RF estimator
    estimator = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
    rfe.fit(X_train, y_train_log)

    # Get selected features
    X_train_rfe = X_train.loc[:, rfe.support_]
    X_test_rfe = X_test.loc[:, rfe.support_]

    # Train full models on selected features
    rf_rfe = RandomForestRegressor(n_estimators=250, max_depth=14, min_samples_split=3,
                                  min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1)
    rf_rfe.fit(X_train_rfe, y_train_log)

    hgb_rfe = HistGradientBoostingRegressor(max_iter=400, max_depth=14, learning_rate=0.05,
                                           min_samples_leaf=4, l2_regularization=0.1, random_state=42,
                                           early_stopping=True, validation_fraction=0.2, n_iter_no_change=20)
    hgb_rfe.fit(X_train_rfe, y_train_log)

    pred_log_rfe = 0.5 * rf_rfe.predict(X_test_rfe) + 0.5 * hgb_rfe.predict(X_test_rfe)
    pred_rfe = np.expm1(pred_log_rfe)

    mae_rfe = mean_absolute_error(y_test_orig, pred_rfe)
    r2_rfe = r2_score(y_test_orig, pred_rfe)

    rfe_results.append({
        'n_features': n_features,
        'mae': mae_rfe,
        'r2': r2_rfe,
        'selected_features': list(X_ultimate.columns[rfe.support_])
    })

    print(f"MAE={mae_rfe:.2f}, R2={r2_rfe:.4f}")

# Results
print("\n" + "="*90)
print("RFE FEATURE SELECTION RESULTS")
print("="*90)
print(f"\n{'N Features':>12} | {'MAE':>8} | {'vs Full':>10} | {'R2':>8}")
print("-" * 60)

print(f"{'Full (86)':>12} | {mae_full:>8.2f} | {'0.00%':>10} | {r2_full:>8.4f}")

for res in rfe_results:
    change = ((mae_full - res['mae']) / mae_full) * 100
    symbol = "+" if change > 0 else ""
    print(f"{res['n_features']:>12} | {res['mae']:>8.2f} | {symbol}{change:>9.2f}% | {res['r2']:>8.4f}")

print("-" * 60)

# Find best
best = min(rfe_results, key=lambda x: x['mae'])
print(f"\n[BEST RFE] {best['n_features']} features")
print(f"   MAE: {best['mae']:.2f}")
print(f"   R2: {best['r2']:.4f}")
improvement = ((mae_full - best['mae']) / mae_full) * 100
print(f"   Improvement vs full: {improvement:+.2f}%")

# Show selected features
print(f"\n[SELECTED FEATURES] Top {best['n_features']} features:")
for i, feat in enumerate(best['selected_features'][:20], 1):  # Show first 20
    print(f"   {i:2}. {feat}")
if len(best['selected_features']) > 20:
    print(f"   ... and {len(best['selected_features']) - 20} more")

# Save
results_df = pd.DataFrame(rfe_results)
results_df.to_csv('experiments/rfe_feature_selection_results.csv', index=False)
print(f"\n[SAVE] Results saved to: experiments/rfe_feature_selection_results.csv")

print("\n" + "="*90)
print("RFE FEATURE SELECTION COMPLETE!")
print("="*90)
print("")
