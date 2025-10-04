#!/usr/bin/env python3
"""
Test Polynomial & Interaction Features
Create non-linear feature combinations for better prediction
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import QuantileTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "="*90)
print(" "*25 + "POLYNOMIAL & INTERACTION FEATURES TEST")
print(" "*15 + "Create Non-Linear Feature Combinations")
print("="*90)

# Load data
baseline_df = pd.read_csv('data/processed/baseline_dataset.csv')
bert_df = pd.read_csv('data/processed/bert_embeddings.csv')
aesthetic_df = pd.read_csv('data/processed/aesthetic_features.csv')
merged_df = baseline_df.merge(aesthetic_df, on='post_id', how='left')

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

# Fill NaN
for col in nima_features:
    merged_df[col] = merged_df[col].fillna(0)

y = merged_df['likes'].copy()

# Experiments
experiments = []

# 1. Baseline: Text + NIMA (current champion)
X_champion = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    merged_df[nima_features].reset_index(drop=True)
], axis=1)
experiments.append({
    'name': 'Champion (Text + NIMA)',
    'features': X_champion
})

# 2. NIMA interactions only (degree 2)
print("\n[FEATURE ENGINEERING] Creating NIMA polynomial features...")
poly_nima = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
nima_poly = poly_nima.fit_transform(merged_df[nima_features])
n_nima_poly = nima_poly.shape[1]
print(f"   NIMA: 8 -> {n_nima_poly} features (degree 2)")

X_nima_poly = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    pd.DataFrame(nima_poly, columns=[f'nima_poly_{i}' for i in range(n_nima_poly)])
], axis=1)
experiments.append({
    'name': f'NIMA Polynomial (degree 2, {n_nima_poly} feats)',
    'features': X_nima_poly
})

# 3. NIMA interactions only (interaction_only=True)
poly_nima_int = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
nima_int = poly_nima_int.fit_transform(merged_df[nima_features])
n_nima_int = nima_int.shape[1]
print(f"   NIMA interactions only: 8 -> {n_nima_int} features")

X_nima_int = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    pd.DataFrame(nima_int, columns=[f'nima_int_{i}' for i in range(n_nima_int)])
], axis=1)
experiments.append({
    'name': f'NIMA Interactions (only, {n_nima_int} feats)',
    'features': X_nima_int
})

# 4. Key baseline features interactions
key_baseline = ['caption_length', 'word_count', 'hashtag_count', 'is_video', 'hour']
poly_baseline = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
baseline_int = poly_baseline.fit_transform(baseline_df[key_baseline])
n_baseline_int = baseline_int.shape[1]
print(f"   Baseline interactions: 5 -> {n_baseline_int} features")

X_baseline_int = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    merged_df[nima_features].reset_index(drop=True),
    pd.DataFrame(baseline_int, columns=[f'base_int_{i}' for i in range(n_baseline_int)])
], axis=1)
experiments.append({
    'name': f'+ Baseline Interactions ({n_baseline_int} feats)',
    'features': X_baseline_int
})

# 5. Manual key interactions (curated based on domain knowledge)
print("\n[MANUAL INTERACTIONS] Creating curated interactions...")
manual_interactions = pd.DataFrame({
    'caption_x_hashtag': baseline_df['caption_length'] * baseline_df['hashtag_count'],
    'caption_x_sharp': baseline_df['caption_length'] * merged_df['aesthetic_sharpness'].fillna(0),
    'video_x_sharp': baseline_df['is_video'] * merged_df['aesthetic_sharpness'].fillna(0),
    'hour_x_weekend': baseline_df['hour'] * baseline_df['is_weekend'],
    'sharp_x_contrast': merged_df['aesthetic_sharpness'].fillna(0) * merged_df['aesthetic_luminance_contrast'].fillna(0),
    'sharp_x_saturation': merged_df['aesthetic_sharpness'].fillna(0) * merged_df['aesthetic_saturation'].fillna(0),
    'saturation_x_brightness': merged_df['aesthetic_saturation'].fillna(0) * merged_df['aesthetic_brightness'].fillna(0),
})
print(f"   Manual interactions: 7 features")

X_manual_int = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    merged_df[nima_features].reset_index(drop=True),
    manual_interactions.reset_index(drop=True)
], axis=1)
experiments.append({
    'name': '+ Manual Interactions (7 feats)',
    'features': X_manual_int
})

# 6. Top NIMA squared (non-linear transforms)
print("\n[NON-LINEAR TRANSFORMS] Creating squared/log features...")
nonlinear_features = pd.DataFrame({
    'sharp_squared': merged_df['aesthetic_sharpness'].fillna(0) ** 2,
    'sharp_sqrt': np.sqrt(merged_df['aesthetic_sharpness'].fillna(0)),
    'saturation_squared': merged_df['aesthetic_saturation'].fillna(0) ** 2,
    'caption_squared': baseline_df['caption_length'] ** 2,
    'caption_log': np.log1p(baseline_df['caption_length']),
})
print(f"   Non-linear transforms: 5 features")

X_nonlinear = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    merged_df[nima_features].reset_index(drop=True),
    nonlinear_features.reset_index(drop=True)
], axis=1)
experiments.append({
    'name': '+ Non-linear Transforms (5 feats)',
    'features': X_nonlinear
})

# 7. ALL interactions combined
X_all_int = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    merged_df[nima_features].reset_index(drop=True),
    manual_interactions.reset_index(drop=True),
    nonlinear_features.reset_index(drop=True)
], axis=1)
experiments.append({
    'name': '+ Manual + Non-linear (12 feats)',
    'features': X_all_int
})

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
print("RESULTS - INTERACTION FEATURES")
print("="*90)
print(f"\n{'Configuration':<50} | {'Feat':>5} | {'MAE':>8} | {'vs Champ':>10} | {'R2':>8}")
print("-" * 95)

for res in results:
    change = ((champion_mae - res['mae']) / champion_mae) * 100
    symbol = "+" if change > 0 else ""
    print(f"{res['name']:<50} | {res['features']:>5} | {res['mae']:>8.2f} | "
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
    print(f"\n   [SUCCESS] Interaction features improved by {improvement:.2f}%!")
else:
    print(f"\n   [INFO] Champion still best. Interaction features don't help.")

# Save
results_df = pd.DataFrame(results)
results_df.to_csv('experiments/interaction_features_results.csv', index=False)
print(f"\n[SAVE] Results saved to: experiments/interaction_features_results.csv")

print("\n" + "="*90)
print("POLYNOMIAL & INTERACTION FEATURES TEST COMPLETE!")
print("="*90)
print("")
