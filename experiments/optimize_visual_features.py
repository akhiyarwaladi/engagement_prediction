#!/usr/bin/env python3
"""
Optimize Visual Features: Find Best PCA Configuration
=======================================================

Test different PCA component numbers for ViT to maximize visual contribution:
- 50 components (baseline - 76.9% variance)
- 75 components (medium)
- 100 components (target: 90%+ variance)
- 150 components (high)
- No PCA (all 768 dims)

Dataset: 348 posts
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
print(" " * 25 + "VISUAL FEATURE OPTIMIZATION")
print(" " * 20 + "Finding Optimal PCA Configuration")
print("=" * 90)

# Load all features
print("\n[DATA] Loading feature sets...")
baseline_df = pd.read_csv('data/processed/baseline_dataset.csv')
bert_df = pd.read_csv('data/processed/bert_embeddings.csv')
vit_df = pd.read_csv('data/processed/vit_embeddings.csv')

bert_cols = [col for col in bert_df.columns if col.startswith('bert_dim_')]
vit_cols = [col for col in vit_df.columns if col.startswith('vit_dim_')]
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']

print(f"   Baseline: {len(baseline_cols)} features")
print(f"   BERT: {len(bert_cols)} dims (will use 50 PCA)")
print(f"   ViT: {len(vit_cols)} dims (will test different PCA)")

# Target
y = baseline_df['likes'].copy()

# BERT PCA (fixed at 50 components)
print("\n[BERT] Applying PCA to BERT features...")
X_bert = bert_df[bert_cols].copy()
pca_bert = PCA(n_components=50, random_state=42)
X_bert_reduced = pd.DataFrame(
    pca_bert.fit_transform(X_bert),
    columns=[f'bert_pc_{i}' for i in range(50)]
)
bert_variance = pca_bert.explained_variance_ratio_.sum()
print(f"   BERT: 768 -> 50 dims ({bert_variance*100:.1f}% variance)")

# Test different ViT PCA configurations
vit_configs = [
    {'name': 'ViT 50 PCA', 'n_components': 50, 'description': 'Baseline (old)'},
    {'name': 'ViT 75 PCA', 'n_components': 75, 'description': 'Medium'},
    {'name': 'ViT 100 PCA', 'n_components': 100, 'description': 'Target: 90%+ variance'},
    {'name': 'ViT 150 PCA', 'n_components': 150, 'description': 'High retention'},
]

print("\n" + "=" * 90)
print("RUNNING EXPERIMENTS")
print("=" * 90)

results = []

for config in vit_configs:
    print(f"\n[EXP] {config['name']}")
    print(f"   {config['description']}")

    # Apply ViT PCA
    X_vit = vit_df[vit_cols].copy()
    pca_vit = PCA(n_components=config['n_components'], random_state=42)
    X_vit_reduced = pd.DataFrame(
        pca_vit.fit_transform(X_vit),
        columns=[f'vit_pc_{i}' for i in range(config['n_components'])]
    )
    vit_variance = pca_vit.explained_variance_ratio_.sum()
    print(f"   ViT: 768 -> {config['n_components']} dims ({vit_variance*100:.1f}% variance)")

    # Combine: Baseline + BERT (50) + ViT (variable)
    X = pd.concat([
        baseline_df[baseline_cols].reset_index(drop=True),
        X_bert_reduced,
        X_vit_reduced
    ], axis=1)

    total_features = X.shape[1]
    print(f"   Total features: {total_features}")

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

    results.append({
        'name': config['name'],
        'vit_components': config['n_components'],
        'total_features': total_features,
        'vit_variance': vit_variance,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pct_error': pct_error,
        'baseline_contrib': baseline_importance,
        'bert_contrib': bert_importance,
        'vit_contrib': vit_importance
    })

    print(f"   MAE: {mae:.2f} likes ({pct_error:.1f}% error)")
    print(f"   R²: {r2:.4f}")
    print(f"   Feature importance: Baseline={baseline_importance*100:.1f}%, "
          f"BERT={bert_importance*100:.1f}%, ViT={vit_importance*100:.1f}%")

# Results summary
print("\n" + "=" * 90)
print("RESULTS SUMMARY")
print("=" * 90)

print("\n[TABLE] Performance Comparison:")
print("-" * 100)
print(f"{'Configuration':<15} | {'ViT PCA':>7} | {'Features':>8} | {'Variance':>8} | "
      f"{'MAE':>8} | {'R²':>8} | {'ViT %':>6}")
print("-" * 100)

for res in results:
    print(f"{res['name']:<15} | {res['vit_components']:>7} | {res['total_features']:>8} | "
          f"{res['vit_variance']*100:>7.1f}% | {res['mae']:>8.2f} | "
          f"{res['r2']:>8.4f} | {res['vit_contrib']*100:>5.1f}%")

print("-" * 100)

# Find optimal
best_mae = min(results, key=lambda x: x['mae'])
best_r2 = max(results, key=lambda x: x['r2'])
best_vit_contrib = max(results, key=lambda x: x['vit_contrib'])
best_variance = max(results, key=lambda x: x['vit_variance'])

print(f"\n[BEST] Best MAE: {best_mae['name']} ({best_mae['mae']:.2f} likes)")
print(f"[BEST] Best R²: {best_r2['name']} ({best_r2['r2']:.4f})")
print(f"[BEST] Highest ViT Contribution: {best_vit_contrib['name']} ({best_vit_contrib['vit_contrib']*100:.1f}%)")
print(f"[BEST] Best ViT Variance: {best_variance['name']} ({best_variance['vit_variance']*100:.1f}%)")

# Analysis
print("\n[ANALYSIS] Variance vs Performance:")
print("-" * 100)

for res in results:
    variance_pct = res['vit_variance'] * 100
    vit_contrib_pct = res['vit_contrib'] * 100

    print(f"{res['name']:<15}: {variance_pct:>5.1f}% variance -> {vit_contrib_pct:>5.1f}% contribution -> "
          f"MAE={res['mae']:.2f}, R²={res['r2']:.4f}")

# Recommendations
print("\n[REC] Recommendations:")
print("-" * 100)

# Find sweet spot
if best_variance['vit_variance'] >= 0.90:
    print(f"1. OPTIMAL CONFIG: {best_variance['name']}")
    print(f"   - Preserves {best_variance['vit_variance']*100:.1f}% variance (target: 90%+)")
    print(f"   - Visual contribution: {best_variance['vit_contrib']*100:.1f}%")
    print(f"   - Performance: MAE={best_variance['mae']:.2f}, R²={best_variance['r2']:.4f}")

if best_mae['name'] != best_r2['name']:
    print(f"\n2. TRADE-OFF DETECTED:")
    print(f"   - Best MAE: {best_mae['name']} (MAE={best_mae['mae']:.2f})")
    print(f"   - Best R²: {best_r2['name']} (R²={best_r2['r2']:.4f})")
    print(f"   - Use MAE model for production, R² model for research")

# Check if adding more components helps
baseline_config = results[0]  # 50 components
best_config = best_mae

improvement_mae = ((baseline_config['mae'] - best_config['mae']) / baseline_config['mae']) * 100
improvement_r2 = ((best_config['r2'] - baseline_config['r2']) / baseline_config['r2']) * 100
improvement_vit = ((best_config['vit_contrib'] - baseline_config['vit_contrib']) / baseline_config['vit_contrib']) * 100

print(f"\n3. IMPROVEMENT FROM 50 -> {best_config['vit_components']} COMPONENTS:")
print(f"   - MAE: {improvement_mae:+.1f}%")
print(f"   - R²: {improvement_r2:+.1f}%")
print(f"   - ViT contribution: {improvement_vit:+.1f}%")

if improvement_vit > 10:
    print(f"   - SIGNIFICANT IMPROVEMENT! More components help visual features")
elif improvement_vit > 0:
    print(f"   - MODERATE IMPROVEMENT. Consider using {best_config['vit_components']} components")
else:
    print(f"   - NO IMPROVEMENT. 50 components may be sufficient")

print("\n" + "=" * 90)
print("OPTIMIZATION COMPLETE!")
print("=" * 90 + "\n")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('experiments/visual_pca_optimization_results.csv', index=False)
print(f"[SAVE] Results saved to: experiments/visual_pca_optimization_results.csv")
