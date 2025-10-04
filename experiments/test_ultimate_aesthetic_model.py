#!/usr/bin/env python3
"""
Ultimate Aesthetic Model Test
NIMA + Composition + Saliency + Color - Find Best Combination
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
print(" "*25 + "ULTIMATE AESTHETIC MODEL TEST")
print(" "*15 + "NIMA + Composition + Saliency + Color - Best Combination")
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

# Feature groups
nima_features = [
    'aesthetic_sharpness', 'aesthetic_noise', 'aesthetic_brightness',
    'aesthetic_exposure_quality', 'aesthetic_color_harmony',
    'aesthetic_saturation', 'aesthetic_saturation_variance',
    'aesthetic_luminance_contrast'
]

composition_features = [
    'composition_rule_of_thirds', 'composition_balance',
    'composition_symmetry', 'composition_edge_density'
]

saliency_features = [
    'saliency_center_bias', 'saliency_attention_spread',
    'saliency_subject_isolation'
]

color_features = [
    'color_vibrancy', 'color_warmth', 'color_diversity'
]

all_aesthetic = nima_features + composition_features + saliency_features + color_features

# Fill NaN
for col in all_aesthetic:
    merged_df[col] = merged_df[col].fillna(0)

# Experiments
experiments = []

# Baseline
X_text = pd.concat([baseline_df[baseline_cols].reset_index(drop=True), X_bert_reduced], axis=1)
experiments.append({'name': 'Text Only', 'features': X_text, 'groups': []})

# NIMA only (best from previous)
X_nima = pd.concat([baseline_df[baseline_cols].reset_index(drop=True), X_bert_reduced,
                    merged_df[nima_features].reset_index(drop=True)], axis=1)
experiments.append({'name': 'NIMA (8)', 'features': X_nima, 'groups': ['NIMA']})

# NIMA + Composition
X_nima_comp = pd.concat([baseline_df[baseline_cols].reset_index(drop=True), X_bert_reduced,
                         merged_df[nima_features + composition_features].reset_index(drop=True)], axis=1)
experiments.append({'name': 'NIMA + Composition (12)', 'features': X_nima_comp, 'groups': ['NIMA', 'Comp']})

# NIMA + Saliency
X_nima_sal = pd.concat([baseline_df[baseline_cols].reset_index(drop=True), X_bert_reduced,
                        merged_df[nima_features + saliency_features].reset_index(drop=True)], axis=1)
experiments.append({'name': 'NIMA + Saliency (11)', 'features': X_nima_sal, 'groups': ['NIMA', 'Sal']})

# NIMA + Color
X_nima_color = pd.concat([baseline_df[baseline_cols].reset_index(drop=True), X_bert_reduced,
                          merged_df[nima_features + color_features].reset_index(drop=True)], axis=1)
experiments.append({'name': 'NIMA + Color (11)', 'features': X_nima_color, 'groups': ['NIMA', 'Color']})

# NIMA + Comp + Sal
X_nima_comp_sal = pd.concat([baseline_df[baseline_cols].reset_index(drop=True), X_bert_reduced,
                              merged_df[nima_features + composition_features + saliency_features].reset_index(drop=True)], axis=1)
experiments.append({'name': 'NIMA + Comp + Sal (15)', 'features': X_nima_comp_sal, 'groups': ['NIMA', 'Comp', 'Sal']})

# NIMA + Comp + Color
X_nima_comp_color = pd.concat([baseline_df[baseline_cols].reset_index(drop=True), X_bert_reduced,
                               merged_df[nima_features + composition_features + color_features].reset_index(drop=True)], axis=1)
experiments.append({'name': 'NIMA + Comp + Color (15)', 'features': X_nima_comp_color, 'groups': ['NIMA', 'Comp', 'Color']})

# NIMA + Sal + Color
X_nima_sal_color = pd.concat([baseline_df[baseline_cols].reset_index(drop=True), X_bert_reduced,
                              merged_df[nima_features + saliency_features + color_features].reset_index(drop=True)], axis=1)
experiments.append({'name': 'NIMA + Sal + Color (14)', 'features': X_nima_sal_color, 'groups': ['NIMA', 'Sal', 'Color']})

# All aesthetic features
X_all = pd.concat([baseline_df[baseline_cols].reset_index(drop=True), X_bert_reduced,
                   merged_df[all_aesthetic].reset_index(drop=True)], axis=1)
experiments.append({'name': 'All Aesthetic (18)', 'features': X_all, 'groups': ['All']})

# Composition only
X_comp = pd.concat([baseline_df[baseline_cols].reset_index(drop=True), X_bert_reduced,
                    merged_df[composition_features].reset_index(drop=True)], axis=1)
experiments.append({'name': 'Composition Only (4)', 'features': X_comp, 'groups': ['Comp']})

# Saliency only
X_sal = pd.concat([baseline_df[baseline_cols].reset_index(drop=True), X_bert_reduced,
                   merged_df[saliency_features].reset_index(drop=True)], axis=1)
experiments.append({'name': 'Saliency Only (3)', 'features': X_sal, 'groups': ['Sal']})

# Color only
X_color = pd.concat([baseline_df[baseline_cols].reset_index(drop=True), X_bert_reduced,
                     merged_df[color_features].reset_index(drop=True)], axis=1)
experiments.append({'name': 'Color Only (3)', 'features': X_color, 'groups': ['Color']})

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
nima_mae = results[1]['mae']

print("\n" + "="*90)
print("RESULTS - ULTIMATE AESTHETIC MODEL")
print("="*90)
print(f"\n{'Configuration':<35} | {'Feat':>5} | {'MAE':>8} | {'vs Text':>9} | {'vs NIMA':>9} | {'R2':>8}")
print("-" * 100)

for res in results:
    change_text = ((baseline_mae - res['mae']) / baseline_mae) * 100
    change_nima = ((nima_mae - res['mae']) / nima_mae) * 100
    symbol1 = "+" if change_text > 0 else ""
    symbol2 = "+" if change_nima > 0 else ""

    print(f"{res['name']:<35} | {res['features']:>5} | {res['mae']:>8.2f} | "
          f"{symbol1}{change_text:>8.2f}% | {symbol2}{change_nima:>8.2f}% | {res['r2']:>8.4f}")

print("-" * 100)

# Find best
best = min(results, key=lambda x: x['mae'])
print(f"\n[BEST] {best['name']}")
print(f"   MAE: {best['mae']:.2f}")
print(f"   R2: {best['r2']:.4f}")
print(f"   Features: {best['features']}")
print(f"   Improvement vs Text: {((baseline_mae-best['mae'])/baseline_mae)*100:+.2f}%")
print(f"   Improvement vs NIMA: {((nima_mae-best['mae'])/nima_mae)*100:+.2f}%")

# Top 3
sorted_results = sorted(results, key=lambda x: x['mae'])
print(f"\n[TOP 3 MODELS]")
for i, res in enumerate(sorted_results[:3], 1):
    improvement = ((baseline_mae - res['mae']) / baseline_mae) * 100
    print(f"   {i}. {res['name']}: MAE={res['mae']:.2f} ({improvement:+.2f}%)")

# Save
results_df = pd.DataFrame(results)
results_df.to_csv('experiments/ultimate_aesthetic_model_results.csv', index=False)
print(f"\n[SAVE] Results saved to: experiments/ultimate_aesthetic_model_results.csv")

print("\n" + "="*90)
print("ULTIMATE AESTHETIC MODEL TEST COMPLETE!")
print("="*90)
print("")
