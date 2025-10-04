#!/usr/bin/env python3
"""
Optimize NIMA Feature Combination
Remove negative features, test best combinations
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
print(" "*25 + "OPTIMIZE NIMA FEATURE COMBINATION")
print(" "*20 + "Remove Negative, Find Optimal Subset")
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
y = merged_df['likes'].copy()

# NIMA features
all_nima = [
    'aesthetic_sharpness',
    'aesthetic_noise',
    'aesthetic_brightness',
    'aesthetic_exposure_quality',
    'aesthetic_color_harmony',  # NEGATIVE
    'aesthetic_saturation',
    'aesthetic_saturation_variance',  # NEGATIVE
    'aesthetic_luminance_contrast'
]

# Top 3 individual
top3 = [
    'aesthetic_sharpness',
    'aesthetic_luminance_contrast',
    'aesthetic_saturation'
]

# Remove negative
positive_only = [
    'aesthetic_sharpness',
    'aesthetic_noise',
    'aesthetic_brightness',
    'aesthetic_exposure_quality',
    'aesthetic_saturation',
    'aesthetic_luminance_contrast'
]

# Fill NaN
for col in all_nima:
    merged_df[col] = merged_df[col].fillna(0)

# Experiments
experiments = []

# Baseline
X_text = pd.concat([baseline_df[baseline_cols].reset_index(drop=True), X_bert_reduced], axis=1)
experiments.append({'name': 'Text Only', 'features': X_text})

# All NIMA (8)
X_all = pd.concat([baseline_df[baseline_cols].reset_index(drop=True), X_bert_reduced,
                   merged_df[all_nima].reset_index(drop=True)], axis=1)
experiments.append({'name': 'All NIMA (8)', 'features': X_all})

# Top 3
X_top3 = pd.concat([baseline_df[baseline_cols].reset_index(drop=True), X_bert_reduced,
                    merged_df[top3].reset_index(drop=True)], axis=1)
experiments.append({'name': 'Top 3 NIMA', 'features': X_top3})

# Positive only (6)
X_positive = pd.concat([baseline_df[baseline_cols].reset_index(drop=True), X_bert_reduced,
                        merged_df[positive_only].reset_index(drop=True)], axis=1)
experiments.append({'name': 'Positive NIMA (6)', 'features': X_positive})

# Top 2
X_top2 = pd.concat([baseline_df[baseline_cols].reset_index(drop=True), X_bert_reduced,
                    merged_df[top3[:2]].reset_index(drop=True)], axis=1)
experiments.append({'name': 'Top 2 (Sharp+Contrast)', 'features': X_top2})

# Sharpness + Saturation
X_sharp_sat = pd.concat([baseline_df[baseline_cols].reset_index(drop=True), X_bert_reduced,
                         merged_df[['aesthetic_sharpness', 'aesthetic_saturation']].reset_index(drop=True)], axis=1)
experiments.append({'name': 'Sharpness + Saturation', 'features': X_sharp_sat})

# Sharpness only
X_sharp = pd.concat([baseline_df[baseline_cols].reset_index(drop=True), X_bert_reduced,
                     merged_df[['aesthetic_sharpness']].reset_index(drop=True)], axis=1)
experiments.append({'name': 'Sharpness Only', 'features': X_sharp})

# Top 4 (add noise)
top4 = top3 + ['aesthetic_noise']
X_top4 = pd.concat([baseline_df[baseline_cols].reset_index(drop=True), X_bert_reduced,
                    merged_df[top4].reset_index(drop=True)], axis=1)
experiments.append({'name': 'Top 4 (add Noise)', 'features': X_top4})

# Top 5 (add brightness)
top5 = top4 + ['aesthetic_brightness']
X_top5 = pd.concat([baseline_df[baseline_cols].reset_index(drop=True), X_bert_reduced,
                    merged_df[top5].reset_index(drop=True)], axis=1)
experiments.append({'name': 'Top 5 (add Brightness)', 'features': X_top5})

print(f"\n[CONFIG] Testing {len(experiments)} combinations")

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
print("RESULTS")
print("="*90)
print(f"\n{'Configuration':<35} | {'Feat':>5} | {'MAE':>8} | {'vs Base':>10} | {'R2':>8}")
print("-" * 90)

for res in results:
    change = ((baseline_mae - res['mae']) / baseline_mae) * 100
    symbol = "+" if change > 0 else ""
    print(f"{res['name']:<35} | {res['features']:>5} | {res['mae']:>8.2f} | {symbol}{change:>9.2f}% | {res['r2']:>8.4f}")

print("-" * 90)

# Find best
best = min(results, key=lambda x: x['mae'])
print(f"\n[BEST] {best['name']}")
print(f"   MAE: {best['mae']:.2f}")
print(f"   R2: {best['r2']:.4f}")
print(f"   Features: {best['features']}")
print(f"   Improvement: {((baseline_mae-best['mae'])/baseline_mae)*100:+.2f}%")

# Save
results_df = pd.DataFrame(results)
results_df.to_csv('experiments/nima_combination_optimization_results.csv', index=False)
print(f"\n[SAVE] Results saved to: experiments/nima_combination_optimization_results.csv")

print("\n" + "="*90)
print("NIMA COMBINATION OPTIMIZATION COMPLETE!")
print("="*90)
print("")
