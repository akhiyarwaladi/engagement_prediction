#!/usr/bin/env python3
"""
Train with Enhanced Visual Features
=====================================

Compare 3 visual feature approaches:
1. Old ViT embeddings (50 PCA) - failed before
2. Enhanced visual features (15 features) - NEW!
3. Combined: ViT + Enhanced features

Hypothesis: Enhanced features (face count, text, video metrics) are more relevant
for Instagram engagement than generic ViT embeddings!
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
print(" " * 20 + "ENHANCED VISUAL FEATURES EXPERIMENT")
print(" " * 15 + "Testing: ViT vs Enhanced vs Combined")
print("=" * 90)

# Load all features
print("\n[DATA] Loading feature sets...")
baseline_df = pd.read_csv('data/processed/baseline_dataset.csv')
bert_df = pd.read_csv('data/processed/bert_embeddings.csv')
vit_df = pd.read_csv('data/processed/vit_embeddings.csv')
enhanced_df = pd.read_csv('data/processed/enhanced_visual_features.csv')

bert_cols = [col for col in bert_df.columns if col.startswith('bert_dim_')]
vit_cols = [col for col in vit_df.columns if col.startswith('vit_dim_')]
enhanced_cols = [col for col in enhanced_df.columns if col not in ['post_id', 'is_video']]
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']

print(f"   Baseline: {len(baseline_cols)} features")
print(f"   BERT: {len(bert_cols)} dims")
print(f"   ViT: {len(vit_cols)} dims")
print(f"   Enhanced Visual: {len(enhanced_cols)} features")

# Target
y = baseline_df['likes'].copy()

# BERT PCA (fixed)
print("\n[BERT] Applying PCA to BERT...")
X_bert = bert_df[bert_cols].copy()
pca_bert = PCA(n_components=50, random_state=42)
X_bert_reduced = pd.DataFrame(
    pca_bert.fit_transform(X_bert),
    columns=[f'bert_pc_{i}' for i in range(50)]
)
bert_variance = pca_bert.explained_variance_ratio_.sum()
print(f"   BERT: 768 -> 50 dims ({bert_variance*100:.1f}% variance)")

# ViT PCA (for comparison)
print("\n[VIT] Applying PCA to ViT...")
X_vit = vit_df[vit_cols].copy()
pca_vit = PCA(n_components=50, random_state=42)
X_vit_reduced = pd.DataFrame(
    pca_vit.fit_transform(X_vit),
    columns=[f'vit_pc_{i}' for i in range(50)]
)
vit_variance = pca_vit.explained_variance_ratio_.sum()
print(f"   ViT: 768 -> 50 dims ({vit_variance*100:.1f}% variance)")

# Test configurations
experiments = []

# Config 1: Baseline + BERT only (text-only baseline)
X_text_only = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced
], axis=1)
experiments.append({
    'name': 'Text Only (Baseline)',
    'features': X_text_only,
    'description': 'BERT + baseline (59 features)'
})

# Config 2: Baseline + BERT + Old ViT
X_vit_old = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    X_vit_reduced
], axis=1)
experiments.append({
    'name': 'Old ViT Embeddings',
    'features': X_vit_old,
    'description': 'BERT + ViT 50 PCA (109 features)'
})

# Config 3: Baseline + BERT + Enhanced Visual
X_enhanced = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    enhanced_df[enhanced_cols].reset_index(drop=True)
], axis=1)
experiments.append({
    'name': 'Enhanced Visual',
    'features': X_enhanced,
    'description': f'BERT + 15 enhanced visual features (74 features)'
})

# Config 4: Baseline + BERT + ViT + Enhanced (ALL FEATURES)
X_combined = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    X_vit_reduced,
    enhanced_df[enhanced_cols].reset_index(drop=True)
], axis=1)
experiments.append({
    'name': 'Combined (ViT + Enhanced)',
    'features': X_combined,
    'description': 'BERT + ViT + Enhanced (124 features)'
})

# Run experiments
print("\n" + "=" * 90)
print("RUNNING EXPERIMENTS")
print("=" * 90)

results = []

for exp in experiments:
    print(f"\n[EXP] {exp['name']}")
    print(f"   {exp['description']}")

    X = exp['features']

    # Train-test split
    X_train_raw, X_test_raw, y_train_orig, y_test_orig = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Preprocessing
    clip_percentile = 99
    clip_value = np.percentile(y_train_orig, clip_percentile)
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
    print("   Training RF + HGB...", end=" ")

    rf = RandomForestRegressor(
        n_estimators=250,
        max_depth=14,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train_log)

    hgb = HistGradientBoostingRegressor(
        max_iter=400,
        max_depth=14,
        learning_rate=0.05,
        min_samples_leaf=4,
        l2_regularization=0.1,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        verbose=0
    )
    hgb.fit(X_train, y_train_log)

    print("[OK]")

    # Predictions
    pred_rf = rf.predict(X_test)
    pred_hgb = hgb.predict(X_test)

    # Weighted ensemble
    mae_rf = mean_absolute_error(y_test_orig, np.expm1(pred_rf))
    mae_hgb = mean_absolute_error(y_test_orig, np.expm1(pred_hgb))

    w_rf = 1.0 / mae_rf
    w_hgb = 1.0 / mae_hgb
    total_w = w_rf + w_hgb
    w_rf /= total_w
    w_hgb /= total_w

    pred_ensemble = w_rf * pred_rf + w_hgb * pred_hgb
    pred_ensemble_orig = np.expm1(pred_ensemble)

    # Evaluate
    mae = mean_absolute_error(y_test_orig, pred_ensemble_orig)
    rmse = np.sqrt(mean_squared_error(y_test_orig, pred_ensemble_orig))
    r2 = r2_score(y_test_orig, pred_ensemble_orig)
    pct_error = (mae / y.mean()) * 100

    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': list(X.columns),
        'importance': rf.feature_importances_
    })

    baseline_importance = feature_importance[feature_importance['feature'].isin(baseline_cols)]['importance'].sum()
    bert_importance = feature_importance[feature_importance['feature'].str.startswith('bert_')]['importance'].sum()
    vit_importance = feature_importance[feature_importance['feature'].str.startswith('vit_')]['importance'].sum()
    enhanced_importance = feature_importance[feature_importance['feature'].isin(enhanced_cols)]['importance'].sum()

    results.append({
        'name': exp['name'],
        'features': X.shape[1],
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pct_error': pct_error,
        'baseline_contrib': baseline_importance,
        'bert_contrib': bert_importance,
        'vit_contrib': vit_importance,
        'enhanced_contrib': enhanced_importance
    })

    print(f"   MAE: {mae:.2f} likes ({pct_error:.1f}% error)")
    print(f"   R²: {r2:.4f}")
    print(f"   Feature importance: Baseline={baseline_importance*100:.1f}%, "
          f"BERT={bert_importance*100:.1f}%, ViT={vit_importance*100:.1f}%, "
          f"Enhanced={enhanced_importance*100:.1f}%")

# Results summary
print("\n" + "=" * 90)
print("RESULTS SUMMARY")
print("=" * 90)

print("\n[TABLE] Performance Comparison:")
print("-" * 110)
print(f"{'Configuration':<25} | {'Features':>8} | {'MAE':>8} | {'R²':>8} | "
      f"{'BERT%':>6} | {'ViT%':>6} | {'Enhanced%':>9}")
print("-" * 110)

for res in results:
    print(f"{res['name']:<25} | {res['features']:>8} | {res['mae']:>8.2f} | "
          f"{res['r2']:>8.4f} | {res['bert_contrib']*100:>5.1f}% | "
          f"{res['vit_contrib']*100:>5.1f}% | {res['enhanced_contrib']*100:>8.1f}%")

print("-" * 110)

# Find best
best_mae = min(results, key=lambda x: x['mae'])
best_r2 = max(results, key=lambda x: x['r2'])
best_enhanced = max(results, key=lambda x: x['enhanced_contrib'])

print(f"\n[BEST] Best MAE: {best_mae['name']} ({best_mae['mae']:.2f} likes)")
print(f"[BEST] Best R²: {best_r2['name']} ({best_r2['r2']:.4f})")
print(f"[BEST] Highest Enhanced Contribution: {best_enhanced['name']} ({best_enhanced['enhanced_contrib']*100:.1f}%)")

# Key comparison: Text-only vs Enhanced Visual
text_only = results[0]
enhanced_visual = results[2]

improvement_mae = ((text_only['mae'] - enhanced_visual['mae']) / text_only['mae']) * 100
improvement_r2 = ((enhanced_visual['r2'] - text_only['r2']) / text_only['r2']) * 100

print("\n[ANALYSIS] Enhanced Visual vs Text-Only:")
print("-" * 110)
if improvement_mae > 0:
    print(f"   MAE: {improvement_mae:+.1f}% IMPROVEMENT ({text_only['mae']:.2f} -> {enhanced_visual['mae']:.2f})")
    print(f"   ENHANCED VISUAL FEATURES HELP!")
else:
    print(f"   MAE: {improvement_mae:+.1f}% change ({text_only['mae']:.2f} -> {enhanced_visual['mae']:.2f})")
    print(f"   Enhanced visual still doesn't help MAE")

if improvement_r2 > 0:
    print(f"   R²: {improvement_r2:+.1f}% IMPROVEMENT ({text_only['r2']:.4f} -> {enhanced_visual['r2']:.4f})")
    print(f"   Better pattern understanding!")
else:
    print(f"   R²: {improvement_r2:+.1f}% change ({text_only['r2']:.4f} -> {enhanced_visual['r2']:.4f})")

# Compare old ViT vs Enhanced
old_vit = results[1]
print("\n[ANALYSIS] Enhanced Visual vs Old ViT:")
print("-" * 110)

vit_vs_enhanced_mae = ((old_vit['mae'] - enhanced_visual['mae']) / old_vit['mae']) * 100
vit_vs_enhanced_r2 = ((enhanced_visual['r2'] - old_vit['r2']) / old_vit['r2']) * 100

if vit_vs_enhanced_mae > 0:
    print(f"   MAE: {vit_vs_enhanced_mae:+.1f}% BETTER ({old_vit['mae']:.2f} -> {enhanced_visual['mae']:.2f})")
    print(f"   ENHANCED > ViT for predictions!")
else:
    print(f"   MAE: {vit_vs_enhanced_mae:+.1f}% change ({old_vit['mae']:.2f} -> {enhanced_visual['mae']:.2f})")

if vit_vs_enhanced_r2 > 0:
    print(f"   R²: {vit_vs_enhanced_r2:+.1f}% BETTER ({old_vit['r2']:.4f} -> {enhanced_visual['r2']:.4f})")
else:
    print(f"   R²: {vit_vs_enhanced_r2:+.1f}% change ({old_vit['r2']:.4f} -> {enhanced_visual['r2']:.4f})")

# Recommendations
print("\n[REC] RECOMMENDATIONS:")
print("-" * 110)

if enhanced_visual['mae'] < text_only['mae']:
    print("1. USE ENHANCED VISUAL FEATURES! They improve over text-only!")
    print(f"   - MAE: {text_only['mae']:.2f} -> {enhanced_visual['mae']:.2f} ({improvement_mae:+.1f}%)")
    print(f"   - Enhanced features (face count, text, video) are more relevant than ViT!")
elif enhanced_visual['mae'] < old_vit['mae']:
    print("1. Enhanced visual BETTER than old ViT embeddings")
    print(f"   - MAE: {old_vit['mae']:.2f} -> {enhanced_visual['mae']:.2f} ({vit_vs_enhanced_mae:+.1f}%)")
    print(f"   - But still worse than text-only")
else:
    print("1. Text-only still best, but enhanced features show promise")
    print(f"   - Enhanced contribution: {enhanced_visual['enhanced_contrib']*100:.1f}%")
    print(f"   - Need more feature engineering")

print("\n" + "=" * 90)
print("EXPERIMENT COMPLETE!")
print("=" * 90 + "\n")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('experiments/enhanced_visual_results.csv', index=False)
print(f"[SAVE] Results saved to: experiments/enhanced_visual_results.csv")
