#!/usr/bin/env python3
"""
Test Individual NIMA Features
Which specific aesthetic feature helps most?
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "="*90)
print(" "*25 + "INDIVIDUAL NIMA FEATURES TEST")
print(" "*20 + "Find the Best Aesthetic Feature(s)")
print("="*90)

# Load data
baseline_df = pd.read_csv('data/processed/baseline_dataset.csv')
bert_df = pd.read_csv('data/processed/bert_embeddings.csv')
aesthetic_df = pd.read_csv('data/processed/aesthetic_features.csv')

# Merge
merged_df = baseline_df.merge(aesthetic_df, on='post_id', how='left')

# BERT PCA
bert_cols = [col for col in bert_df.columns if col.startswith('bert_dim_')]
pca_bert = PCA(n_components=50, random_state=42)
X_bert_reduced = pd.DataFrame(
    pca_bert.fit_transform(bert_df[bert_cols]),
    columns=[f'bert_pc_{i}' for i in range(50)]
)

# Baseline features
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']

# Target
y = merged_df['likes'].copy()

# NIMA features
nima_features = [
    'aesthetic_sharpness',
    'aesthetic_noise',
    'aesthetic_brightness',
    'aesthetic_exposure_quality',
    'aesthetic_color_harmony',
    'aesthetic_saturation',
    'aesthetic_saturation_variance',
    'aesthetic_luminance_contrast'
]

# Fill NaN
for col in nima_features:
    merged_df[col] = merged_df[col].fillna(0)

# Experiments
experiments = []

# Baseline
X_text = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced
], axis=1)
experiments.append({
    'name': 'Text Only',
    'features': X_text,
    'visual_cols': []
})

# Individual NIMA features
for feat in nima_features:
    X_temp = pd.concat([
        baseline_df[baseline_cols].reset_index(drop=True),
        X_bert_reduced,
        merged_df[[feat]].reset_index(drop=True)
    ], axis=1)
    experiments.append({
        'name': f'Text + {feat.replace("aesthetic_", "").title()}',
        'features': X_temp,
        'visual_cols': [feat]
    })

# All NIMA
X_all = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    merged_df[nima_features].reset_index(drop=True)
], axis=1)
experiments.append({
    'name': 'Text + All NIMA (8)',
    'features': X_all,
    'visual_cols': nima_features
})

print(f"\n[CONFIG] Testing {len(experiments)} configurations")

# Run experiments
results = []

for exp in experiments:
    X = exp['features']

    # Train-test split
    X_train_raw, X_test_raw, y_train_orig, y_test_orig = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessing
    clip_value = np.percentile(y_train_orig, 99)
    y_train_clipped = np.clip(y_train_orig, None, clip_value)
    y_train_log = np.log1p(y_train_clipped)
    y_test_log = np.log1p(y_test_orig)

    transformer = QuantileTransformer(
        n_quantiles=min(100, len(X_train_raw)),
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

    # Ensemble
    rf = RandomForestRegressor(
        n_estimators=250, max_depth=14, min_samples_split=3,
        min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train_log)

    hgb = HistGradientBoostingRegressor(
        max_iter=400, max_depth=14, learning_rate=0.05, min_samples_leaf=4,
        l2_regularization=0.1, random_state=42, early_stopping=True,
        validation_fraction=0.2, n_iter_no_change=20
    )
    hgb.fit(X_train, y_train_log)

    # Predictions
    pred_log = 0.5 * rf.predict(X_test) + 0.5 * hgb.predict(X_test)
    pred = np.expm1(pred_log)

    # Evaluate
    mae = mean_absolute_error(y_test_orig, pred)
    r2 = r2_score(y_test_orig, pred)

    # Feature importance
    visual_importance = 0.0
    for col in exp['visual_cols']:
        idx = list(X.columns).index(col)
        visual_importance += rf.feature_importances_[idx]

    results.append({
        'name': exp['name'],
        'mae': mae,
        'r2': r2,
        'visual_importance': visual_importance
    })

# Results
baseline_mae = results[0]['mae']

print("\n" + "="*90)
print("RESULTS")
print("="*90)
print(f"\n{'Feature':<40} | {'MAE':>8} | {'vs Base':>10} | {'R2':>8} | {'Importance':>10}")
print("-" * 90)

for res in results:
    change = ((baseline_mae - res['mae']) / baseline_mae) * 100
    symbol = "+" if change > 0 else ""

    print(f"{res['name']:<40} | {res['mae']:>8.2f} | {symbol}{change:>9.2f}% | "
          f"{res['r2']:>8.4f} | {res['visual_importance']*100:>9.2f}%")

print("-" * 90)

# Find best single feature
single_features = results[1:-1]  # exclude baseline and all NIMA
best_single = min(single_features, key=lambda x: x['mae'])

print(f"\n[BEST SINGLE] {best_single['name']}")
print(f"   MAE: {best_single['mae']:.2f}")
print(f"   Improvement: {((baseline_mae-best_single['mae'])/baseline_mae)*100:+.2f}%")

# Top 3 features
sorted_features = sorted(single_features, key=lambda x: x['mae'])
print(f"\n[TOP 3 FEATURES]")
for i, feat in enumerate(sorted_features[:3], 1):
    improvement = ((baseline_mae - feat['mae']) / baseline_mae) * 100
    print(f"   {i}. {feat['name']}: MAE={feat['mae']:.2f} ({improvement:+.2f}%)")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('experiments/individual_nima_features_results.csv', index=False)
print(f"\n[SAVE] Results saved to: experiments/individual_nima_features_results.csv")

print("\n" + "="*90)
print("")
