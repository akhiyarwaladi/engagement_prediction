#!/usr/bin/env python3
"""
Test Individual Quality Features - Deep Dive
=============================================

We know quality features help (+0.11% MAE). But which one matters MOST?
Test each quality feature individually:
1. Sharpness only (1.45% importance)
2. Contrast only (1.02% importance)
3. Aspect ratio only (0.74% importance)

Also test all possible combinations!
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

print("\n" + "=" * 90)
print(" " * 15 + "INDIVIDUAL QUALITY FEATURES DEEP DIVE")
print(" " * 10 + "Which Quality Feature Matters Most?")
print("=" * 90)

# Load features
print("\n[DATA] Loading feature sets...")
baseline_df = pd.read_csv('data/processed/baseline_dataset.csv')
bert_df = pd.read_csv('data/processed/bert_embeddings.csv')
enhanced_df = pd.read_csv('data/processed/enhanced_visual_features.csv')

bert_cols = [col for col in bert_df.columns if col.startswith('bert_dim_')]
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']

y = baseline_df['likes'].copy()

# BERT PCA
print("\n[BERT] Applying PCA...")
X_bert = bert_df[bert_cols].copy()
pca_bert = PCA(n_components=50, random_state=42)
X_bert_reduced = pd.DataFrame(
    pca_bert.fit_transform(X_bert),
    columns=[f'bert_pc_{i}' for i in range(50)]
)

# Test configurations - ALL possible quality feature combinations
experiments = []

# Baseline
X_baseline = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced
], axis=1)
experiments.append({
    'name': 'Text-Only (Baseline)',
    'features': X_baseline,
    'quality_features': []
})

# Individual features
quality_individual = [
    ['sharpness'],
    ['contrast'],
    ['aspect_ratio']
]

for qf in quality_individual:
    X_temp = pd.concat([
        baseline_df[baseline_cols].reset_index(drop=True),
        X_bert_reduced,
        enhanced_df[qf].reset_index(drop=True)
    ], axis=1)
    experiments.append({
        'name': f'{qf[0].title()} Only',
        'features': X_temp,
        'quality_features': qf
    })

# Pairs
quality_pairs = [
    ['sharpness', 'contrast'],
    ['sharpness', 'aspect_ratio'],
    ['contrast', 'aspect_ratio']
]

for qf in quality_pairs:
    X_temp = pd.concat([
        baseline_df[baseline_cols].reset_index(drop=True),
        X_bert_reduced,
        enhanced_df[qf].reset_index(drop=True)
    ], axis=1)
    experiments.append({
        'name': f'{qf[0].title()} + {qf[1].title()}',
        'features': X_temp,
        'quality_features': qf
    })

# All three
X_all_quality = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    enhanced_df[['sharpness', 'contrast', 'aspect_ratio']].reset_index(drop=True)
], axis=1)
experiments.append({
    'name': 'All Quality Features',
    'features': X_all_quality,
    'quality_features': ['sharpness', 'contrast', 'aspect_ratio']
})

print(f"\n[CONFIG] Testing {len(experiments)} configurations:")
for i, exp in enumerate(experiments, 1):
    print(f"   {i}. {exp['name']} ({exp['features'].shape[1]} features)")

# Run experiments
print("\n" + "=" * 90)
print("RUNNING EXPERIMENTS")
print("=" * 90)

results = []

for exp in experiments:
    print(f"\n[TEST] {exp['name']}")
    X = exp['features']

    # Train-test split
    X_train_raw, X_test_raw, y_train_orig, y_test_orig = train_test_split(
        X, y, test_size=0.3, random_state=42
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

    # Train ensemble
    print("   Training...", end=" ")

    rf = RandomForestRegressor(
        n_estimators=250, max_depth=14, min_samples_split=3,
        min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train_log)

    hgb = HistGradientBoostingRegressor(
        max_iter=400, max_depth=14, learning_rate=0.05,
        min_samples_leaf=4, l2_regularization=0.1, random_state=42,
        early_stopping=True, validation_fraction=0.2, n_iter_no_change=20, verbose=0
    )
    hgb.fit(X_train, y_train_log)

    # Predictions
    pred_rf = np.expm1(rf.predict(X_test))
    pred_hgb = np.expm1(hgb.predict(X_test))

    mae_rf = mean_absolute_error(y_test_orig, pred_rf)
    mae_hgb = mean_absolute_error(y_test_orig, pred_hgb)

    w_rf = 1.0 / mae_rf
    w_hgb = 1.0 / mae_hgb
    total_w = w_rf + w_hgb
    w_rf /= total_w
    w_hgb /= total_w

    pred_ensemble_log = w_rf * rf.predict(X_test) + w_hgb * hgb.predict(X_test)
    pred_ensemble = np.expm1(pred_ensemble_log)

    # Evaluate
    mae = mean_absolute_error(y_test_orig, pred_ensemble)
    rmse = np.sqrt(mean_squared_error(y_test_orig, pred_ensemble))
    r2 = r2_score(y_test_orig, pred_ensemble)

    # Feature importance for quality features
    feature_importance = pd.DataFrame({
        'feature': list(X.columns),
        'importance': rf.feature_importances_
    })

    quality_importance = 0.0
    quality_breakdown = {}
    for qf in exp['quality_features']:
        qf_imp = feature_importance[feature_importance['feature'] == qf]['importance'].values
        if len(qf_imp) > 0:
            quality_breakdown[qf] = qf_imp[0]
            quality_importance += qf_imp[0]

    results.append({
        'name': exp['name'],
        'features': X.shape[1],
        'quality_features': ', '.join(exp['quality_features']) if exp['quality_features'] else 'None',
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'quality_importance': quality_importance,
        'quality_breakdown': quality_breakdown
    })

    print(f"MAE={mae:.2f}, R2={r2:.4f}, Quality={quality_importance*100:.2f}%")

# Results summary
print("\n" + "=" * 90)
print("RESULTS SUMMARY")
print("=" * 90)

baseline_mae = results[0]['mae']
baseline_r2 = results[0]['r2']

print("\n[TABLE] Performance Comparison:")
print("-" * 120)
print(f"{'Configuration':<30} | {'Features':>8} | {'Quality Feat':>20} | {'MAE':>8} | {'R2':>8} | {'vs Base MAE':>12} | {'Quality %':>10}")
print("-" * 120)

for res in results:
    mae_change = ((baseline_mae - res['mae']) / baseline_mae) * 100
    symbol = "+" if mae_change > 0 else ""

    print(f"{res['name']:<30} | {res['features']:>8} | {res['quality_features']:>20} | "
          f"{res['mae']:>8.2f} | {res['r2']:>8.4f} | {symbol}{mae_change:>11.2f}% | "
          f"{res['quality_importance']*100:>9.2f}%")

print("-" * 120)

# Find best
best_mae = min(results[1:], key=lambda x: x['mae'])  # Skip baseline
best_r2 = max(results[1:], key=lambda x: x['r2'])

print(f"\n[BEST] Best MAE: {best_mae['name']} (MAE={best_mae['mae']:.2f})")
print(f"[BEST] Best R2: {best_r2['name']} (R2={best_r2['r2']:.4f})")

# Analysis
print("\n" + "=" * 90)
print("ANALYSIS - WHICH QUALITY FEATURE MATTERS MOST?")
print("=" * 90)

# Individual features comparison
individual_results = results[1:4]  # Sharpness, Contrast, Aspect ratio
individual_sorted = sorted(individual_results, key=lambda x: x['mae'])

print("\n[RANKING] Individual Quality Features (Best to Worst):")
for i, res in enumerate(individual_sorted, 1):
    mae_improve = ((baseline_mae - res['mae']) / baseline_mae) * 100
    symbol = "+" if mae_improve > 0 else ""
    qf_name = res['quality_features']

    print(f"   {i}. {qf_name:15} MAE={res['mae']:.2f} ({symbol}{mae_improve:.2f}% vs baseline), "
          f"Importance={res['quality_importance']*100:.2f}%")

# Best individual
best_individual = individual_sorted[0]
print(f"\n[WINNER] Best SINGLE quality feature: {best_individual['quality_features']}")
print(f"   MAE improvement: {((baseline_mae - best_individual['mae'])/baseline_mae)*100:+.2f}%")
print(f"   R2 improvement: {((best_individual['r2'] - baseline_r2)/baseline_r2)*100:+.2f}%")

# Check if combining helps
all_quality = results[-1]
print(f"\n[COMBINATION] All 3 quality features combined:")
print(f"   MAE: {all_quality['mae']:.2f}")
print(f"   Best individual: {best_individual['mae']:.2f}")

if all_quality['mae'] < best_individual['mae']:
    improvement = ((best_individual['mae'] - all_quality['mae']) / best_individual['mae']) * 100
    print(f"   Combining HELPS! (+{improvement:.2f}% better than best individual)")
else:
    degradation = ((all_quality['mae'] - best_individual['mae']) / best_individual['mae']) * 100
    print(f"   Combining HURTS! (-{degradation:.2f}% worse than best individual)")

# Pair analysis
print(f"\n[PAIRS] Best pairs:")
pair_results = results[4:7]
pair_sorted = sorted(pair_results, key=lambda x: x['mae'])

for i, res in enumerate(pair_sorted, 1):
    mae_improve = ((baseline_mae - res['mae']) / baseline_mae) * 100
    print(f"   {i}. {res['quality_features']:30} MAE={res['mae']:.2f} ({mae_improve:+.2f}%)")

# Feature importance breakdown
print(f"\n[IMPORTANCE] Quality Feature Importance Breakdown:")
print("-" * 60)

# Collect all importance values
importance_summary = {}
for res in results[1:]:  # Skip baseline
    for qf, imp in res['quality_breakdown'].items():
        if qf not in importance_summary:
            importance_summary[qf] = []
        importance_summary[qf].append(imp * 100)

for qf in ['sharpness', 'contrast', 'aspect_ratio']:
    if qf in importance_summary:
        importances = importance_summary[qf]
        avg_imp = np.mean(importances)
        min_imp = np.min(importances)
        max_imp = np.max(importances)
        print(f"   {qf:15} avg={avg_imp:.2f}%, min={min_imp:.2f}%, max={max_imp:.2f}%")

# Recommendations
print("\n" + "=" * 90)
print("RECOMMENDATIONS")
print("=" * 90)

print(f"\n1. BEST SINGLE FEATURE:")
print(f"   Use: {best_individual['quality_features']}")
print(f"   Performance: MAE={best_individual['mae']:.2f} ({((baseline_mae-best_individual['mae'])/baseline_mae)*100:+.2f}%)")

print(f"\n2. OPTIMAL CONFIGURATION:")
if all_quality['mae'] < baseline_mae:
    print(f"   Use: All 3 quality features (sharpness + contrast + aspect_ratio)")
    print(f"   Performance: MAE={all_quality['mae']:.2f} ({((baseline_mae-all_quality['mae'])/baseline_mae)*100:+.2f}%)")
    print(f"   Justification: Combining features helps!")
else:
    print(f"   Stick with text-only baseline")
    print(f"   Quality features add noise, not signal")

print(f"\n3. FEATURE PRIORITY:")
print(f"   If limited to 1 feature: {individual_sorted[0]['quality_features']}")
print(f"   If using 2 features: {pair_sorted[0]['quality_features']}")
print(f"   If using all 3: sharpness + contrast + aspect_ratio")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('experiments/individual_quality_features_results.csv', index=False)
print(f"\n[SAVE] Results saved to: experiments/individual_quality_features_results.csv")

print("\n" + "=" * 90)
print("EXPERIMENT COMPLETE!")
print("=" * 90 + "\n")
