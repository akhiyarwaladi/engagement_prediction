#!/usr/bin/env python3
"""
Combine NIMA with Previous Best Features
NIMA + Aspect Ratio + Contrast = Ultimate Model?
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
print(" "*25 + "NIMA + PREVIOUS BEST FEATURES")
print(" "*20 + "Can we beat the current champion?")
print("="*90)

# Load data
baseline_df = pd.read_csv('data/processed/baseline_dataset.csv')
bert_df = pd.read_csv('data/processed/bert_embeddings.csv')
aesthetic_df = pd.read_csv('data/processed/aesthetic_features.csv')

# Try to load enhanced features (previous best)
try:
    enhanced_df = pd.read_csv('data/processed/enhanced_visual_features.csv')
    has_enhanced = True
except:
    print("[WARN] Enhanced features not found")
    has_enhanced = False

merged_df = baseline_df.merge(aesthetic_df, on='post_id', how='left')
if has_enhanced:
    merged_df = merged_df.merge(
        enhanced_df[['post_id', 'contrast', 'aspect_ratio']],
        on='post_id',
        how='left'
    )

# BERT PCA
bert_cols = [col for col in bert_df.columns if col.startswith('bert_dim_')]
pca_bert = PCA(n_components=50, random_state=42)
X_bert_reduced = pd.DataFrame(
    pca_bert.fit_transform(bert_df[bert_cols]),
    columns=[f'bert_pc_{i}' for i in range(50)]
)

baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']
y = merged_df['likes'].copy()

# NIMA features (best 3 from individual tests)
nima_top3 = [
    'aesthetic_sharpness',
    'aesthetic_luminance_contrast',
    'aesthetic_saturation'
]

# All NIMA
nima_all = [
    'aesthetic_sharpness', 'aesthetic_noise', 'aesthetic_brightness',
    'aesthetic_exposure_quality', 'aesthetic_color_harmony',
    'aesthetic_saturation', 'aesthetic_saturation_variance',
    'aesthetic_luminance_contrast'
]

# Fill NaN
for col in nima_all:
    merged_df[col] = merged_df[col].fillna(0)

if has_enhanced:
    merged_df['contrast'] = merged_df['contrast'].fillna(0)
    merged_df['aspect_ratio'] = merged_df['aspect_ratio'].fillna(0)

# Experiments
experiments = []

# Baseline
X_text = pd.concat([baseline_df[baseline_cols].reset_index(drop=True), X_bert_reduced], axis=1)
experiments.append({'name': 'Text Only', 'features': X_text})

# Previous best
if has_enhanced:
    X_prev_best = pd.concat([
        baseline_df[baseline_cols].reset_index(drop=True),
        X_bert_reduced,
        merged_df[['contrast', 'aspect_ratio']].reset_index(drop=True)
    ], axis=1)
    experiments.append({'name': 'Previous Best (Contrast+Aspect)', 'features': X_prev_best})

# NIMA only
X_nima = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    merged_df[nima_all].reset_index(drop=True)
], axis=1)
experiments.append({'name': 'NIMA (8)', 'features': X_nima})

# NIMA Top 3
X_nima_top3 = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    merged_df[nima_top3].reset_index(drop=True)
], axis=1)
experiments.append({'name': 'NIMA Top 3', 'features': X_nima_top3})

# NIMA + Previous Best
if has_enhanced:
    X_nima_prev = pd.concat([
        baseline_df[baseline_cols].reset_index(drop=True),
        X_bert_reduced,
        merged_df[nima_all + ['contrast', 'aspect_ratio']].reset_index(drop=True)
    ], axis=1)
    experiments.append({'name': 'NIMA + Contrast + Aspect (10)', 'features': X_nima_prev})

    # NIMA Top 3 + Previous Best
    X_nima_top3_prev = pd.concat([
        baseline_df[baseline_cols].reset_index(drop=True),
        X_bert_reduced,
        merged_df[nima_top3 + ['contrast', 'aspect_ratio']].reset_index(drop=True)
    ], axis=1)
    experiments.append({'name': 'NIMA Top3 + Contrast + Aspect (5)', 'features': X_nima_top3_prev})

    # Just Sharpness + Previous Best
    X_sharp_prev = pd.concat([
        baseline_df[baseline_cols].reset_index(drop=True),
        X_bert_reduced,
        merged_df[['aesthetic_sharpness', 'contrast', 'aspect_ratio']].reset_index(drop=True)
    ], axis=1)
    experiments.append({'name': 'Sharpness + Contrast + Aspect (3)', 'features': X_sharp_prev})

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
baseline_mae = results[0]['mae']

print("\n" + "="*90)
print("RESULTS - NIMA + PREVIOUS BEST")
print("="*90)
print(f"\n{'Configuration':<45} | {'Feat':>5} | {'MAE':>8} | {'vs Base':>10} | {'R2':>8}")
print("-" * 95)

for res in results:
    change = ((baseline_mae - res['mae']) / baseline_mae) * 100
    symbol = "+" if change > 0 else ""
    print(f"{res['name']:<45} | {res['features']:>5} | {res['mae']:>8.2f} | "
          f"{symbol}{change:>9.2f}% | {res['r2']:>8.4f}")

print("-" * 95)

# Find best
best = min(results, key=lambda x: x['mae'])
print(f"\n[CHAMPION] {best['name']}")
print(f"   MAE: {best['mae']:.2f}")
print(f"   R2: {best['r2']:.4f}")
print(f"   Features: {best['features']}")
print(f"   Improvement: {((baseline_mae-best['mae'])/baseline_mae)*100:+.2f}%")

# Compare to previous best
if has_enhanced and len(results) >= 2:
    prev_best_mae = results[1]['mae']
    champion_mae = best['mae']
    improvement = ((prev_best_mae - champion_mae) / prev_best_mae) * 100

    print(f"\n[VS PREVIOUS BEST]")
    print(f"   Previous Best MAE: {prev_best_mae:.2f}")
    print(f"   Champion MAE: {champion_mae:.2f}")
    print(f"   Improvement: {improvement:+.2f}%")

    if improvement > 0:
        print(f"   [SUCCESS] New champion found! {improvement:.2f}% better!")
    else:
        print(f"   [INFO] Previous best still holds")

# Save
results_df = pd.DataFrame(results)
results_df.to_csv('experiments/nima_plus_previous_best_results.csv', index=False)
print(f"\n[SAVE] Results saved to: experiments/nima_plus_previous_best_results.csv")

print("\n" + "="*90)
print("NIMA + PREVIOUS BEST TEST COMPLETE!")
print("="*90)
print("")
