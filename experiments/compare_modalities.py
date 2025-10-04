#!/usr/bin/env python3
"""
Experimental Comparison: Different Modality Combinations
===========================================================

Compare performance of:
1. Baseline only (9 features)
2. Baseline + BERT (text-only)
3. Baseline + ViT (visual-only)
4. Baseline + BERT + ViT (multimodal)

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

print("\n" + "=" * 80)
print(" " * 20 + "MODALITY COMPARISON EXPERIMENTS")
print(" " * 25 + "348 Posts Dataset")
print("=" * 80)

# Load all features
print("\n[DATA] Loading all feature sets...")
baseline_df = pd.read_csv('data/processed/baseline_dataset.csv')
bert_df = pd.read_csv('data/processed/bert_embeddings.csv')
vit_df = pd.read_csv('data/processed/vit_embeddings.csv')

bert_cols = [col for col in bert_df.columns if col.startswith('bert_dim_')]
vit_cols = [col for col in vit_df.columns if col.startswith('vit_dim_')]
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']

print(f"   Baseline: {len(baseline_cols)} features")
print(f"   BERT: {len(bert_cols)} features")
print(f"   ViT: {len(vit_cols)} features")

# Target
y = baseline_df['likes'].copy()
print(f"\n[STATS] Dataset: {len(y)} posts")
print(f"   Mean likes: {y.mean():.1f}")
print(f"   Std likes: {y.std():.1f}")
print(f"   Max likes: {y.max():.0f}")

# Define experiments
experiments = []

# Experiment 1: Baseline only
experiments.append({
    'name': 'Baseline Only',
    'features': baseline_df[baseline_cols].copy(),
    'description': 'Simple temporal and metadata features (9 features)'
})

# Experiment 2: Text-only (Baseline + BERT)
X_bert = bert_df[bert_cols].copy()
pca_bert = PCA(n_components=50, random_state=42)
X_bert_reduced = pd.DataFrame(
    pca_bert.fit_transform(X_bert),
    columns=[f'bert_pc_{i}' for i in range(50)]
)
bert_variance = pca_bert.explained_variance_ratio_.sum()

X_text = pd.concat([baseline_df[baseline_cols].reset_index(drop=True),
                    X_bert_reduced], axis=1)
experiments.append({
    'name': 'Text Only',
    'features': X_text,
    'description': f'Baseline + IndoBERT (59 features, {bert_variance*100:.1f}% variance)'
})

# Experiment 3: Visual-only (Baseline + ViT)
X_vit = vit_df[vit_cols].copy()
pca_vit = PCA(n_components=50, random_state=42)
X_vit_reduced = pd.DataFrame(
    pca_vit.fit_transform(X_vit),
    columns=[f'vit_pc_{i}' for i in range(50)]
)
vit_variance = pca_vit.explained_variance_ratio_.sum()

X_visual = pd.concat([baseline_df[baseline_cols].reset_index(drop=True),
                      X_vit_reduced], axis=1)
experiments.append({
    'name': 'Visual Only',
    'features': X_visual,
    'description': f'Baseline + ViT (59 features, {vit_variance*100:.1f}% variance)'
})

# Experiment 4: Multimodal (Baseline + BERT + ViT)
X_multimodal = pd.concat([baseline_df[baseline_cols].reset_index(drop=True),
                          X_bert_reduced, X_vit_reduced], axis=1)
experiments.append({
    'name': 'Multimodal',
    'features': X_multimodal,
    'description': f'Baseline + BERT + ViT (109 features)'
})

# Run experiments
print("\n" + "=" * 80)
print("RUNNING EXPERIMENTS")
print("=" * 80)

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
    print("   Training Random Forest...", end=" ")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train_log)
    print("[OK]")

    print("   Training HistGradientBoosting...", end=" ")
    hgb = HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=12,
        learning_rate=0.05,
        min_samples_leaf=5,
        l2_regularization=0.1,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=15,
        verbose=0
    )
    hgb.fit(X_train, y_train_log)
    print("[OK]")

    # Predictions
    pred_rf = rf.predict(X_test)
    pred_hgb = hgb.predict(X_test)

    # MAE-weighted ensemble
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

    # Percentage error
    pct_error = (mae / y.mean()) * 100

    results.append({
        'name': exp['name'],
        'features': X.shape[1],
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pct_error': pct_error,
        'rf_weight': w_rf,
        'hgb_weight': w_hgb
    })

    print(f"   MAE: {mae:.2f} likes ({pct_error:.1f}% error)")
    print(f"   RMSE: {rmse:.2f} likes")
    print(f"   R²: {r2:.4f}")

# Results summary
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print("\n[TABLE] Performance Comparison:")
print("-" * 90)
print(f"{'Experiment':<20} | {'Features':>8} | {'MAE':>8} | {'RMSE':>8} | {'R²':>8} | {'% Error':>8}")
print("-" * 90)

for res in results:
    print(f"{res['name']:<20} | {res['features']:>8} | {res['mae']:>8.2f} | "
          f"{res['rmse']:>8.2f} | {res['r2']:>8.4f} | {res['pct_error']:>7.1f}%")

print("-" * 90)

# Find best models
best_mae = min(results, key=lambda x: x['mae'])
best_r2 = max(results, key=lambda x: x['r2'])
best_pct = min(results, key=lambda x: x['pct_error'])

print(f"\n[BEST] Best MAE: {best_mae['name']} ({best_mae['mae']:.2f} likes)")
print(f"[BEST] Best R²: {best_r2['name']} ({best_r2['r2']:.4f})")
print(f"[BEST] Best % Error: {best_pct['name']} ({best_pct['pct_error']:.1f}%)")

# Improvements
baseline_result = results[0]
text_result = results[1]
visual_result = results[2]
multimodal_result = results[3]

print("\n[ANALYSIS] Contribution Analysis:")
print("-" * 80)

text_improvement_mae = ((baseline_result['mae'] - text_result['mae']) / baseline_result['mae']) * 100
text_improvement_r2 = ((text_result['r2'] - baseline_result['r2']) / baseline_result['r2']) * 100

visual_improvement_mae = ((baseline_result['mae'] - visual_result['mae']) / baseline_result['mae']) * 100
visual_improvement_r2 = ((visual_result['r2'] - baseline_result['r2']) / baseline_result['r2']) * 100

multimodal_improvement_mae = ((baseline_result['mae'] - multimodal_result['mae']) / baseline_result['mae']) * 100
multimodal_improvement_r2 = ((multimodal_result['r2'] - baseline_result['r2']) / baseline_result['r2']) * 100

print(f"\nText-only (vs Baseline):")
print(f"   MAE: {text_improvement_mae:+.1f}% ({baseline_result['mae']:.2f} -> {text_result['mae']:.2f})")
print(f"   R²: {text_improvement_r2:+.1f}% ({baseline_result['r2']:.4f} -> {text_result['r2']:.4f})")

print(f"\nVisual-only (vs Baseline):")
print(f"   MAE: {visual_improvement_mae:+.1f}% ({baseline_result['mae']:.2f} -> {visual_result['mae']:.2f})")
print(f"   R²: {visual_improvement_r2:+.1f}% ({baseline_result['r2']:.4f} -> {visual_result['r2']:.4f})")

print(f"\nMultimodal (vs Baseline):")
print(f"   MAE: {multimodal_improvement_mae:+.1f}% ({baseline_result['mae']:.2f} -> {multimodal_result['mae']:.2f})")
print(f"   R²: {multimodal_improvement_r2:+.1f}% ({baseline_result['r2']:.4f} -> {multimodal_result['r2']:.4f})")

# Modality synergy analysis
print("\n[SYNERGY] Multimodal vs Individual Modalities:")
print("-" * 80)

text_vs_multi_mae = ((text_result['mae'] - multimodal_result['mae']) / text_result['mae']) * 100
text_vs_multi_r2 = ((multimodal_result['r2'] - text_result['r2']) / text_result['r2']) * 100

visual_vs_multi_mae = ((visual_result['mae'] - multimodal_result['mae']) / visual_result['mae']) * 100
visual_vs_multi_r2 = ((multimodal_result['r2'] - visual_result['r2']) / visual_result['r2']) * 100

print(f"\nMultimodal vs Text-only:")
print(f"   MAE: {text_vs_multi_mae:+.1f}% ({text_result['mae']:.2f} -> {multimodal_result['mae']:.2f})")
print(f"   R²: {text_vs_multi_r2:+.1f}% ({text_result['r2']:.4f} -> {multimodal_result['r2']:.4f})")

print(f"\nMultimodal vs Visual-only:")
print(f"   MAE: {visual_vs_multi_mae:+.1f}% ({visual_result['mae']:.2f} -> {multimodal_result['mae']:.2f})")
print(f"   R²: {visual_vs_multi_r2:+.1f}% ({visual_result['r2']:.4f} -> {multimodal_result['r2']:.4f})")

# Key findings
print("\n[KEY] Key Findings:")
print("-" * 80)

if text_result['mae'] < visual_result['mae']:
    print("1. Text features MORE important than visual for MAE")
    diff_pct = ((visual_result['mae'] - text_result['mae']) / visual_result['mae']) * 100
    print(f"   Text {diff_pct:.1f}% better than Visual")
else:
    print("1. Visual features MORE important than text for MAE")
    diff_pct = ((text_result['mae'] - visual_result['mae']) / text_result['mae']) * 100
    print(f"   Visual {diff_pct:.1f}% better than Text")

if text_result['r2'] > visual_result['r2']:
    print("\n2. Text features MORE important than visual for R²")
    diff_pct = ((text_result['r2'] - visual_result['r2']) / visual_result['r2']) * 100
    print(f"   Text {diff_pct:.1f}% better than Visual")
else:
    print("\n2. Visual features MORE important than text for R²")
    diff_pct = ((visual_result['r2'] - text_result['r2']) / text_result['r2']) * 100
    print(f"   Visual {diff_pct:.1f}% better than Text")

if multimodal_result['r2'] > max(text_result['r2'], visual_result['r2']):
    print("\n3. Multimodal fusion BENEFICIAL (R² improved)")
    print("   Combining modalities provides synergy")
else:
    print("\n3. Multimodal fusion NOT BENEFICIAL (R² decreased)")
    print("   Individual modalities may be sufficient")

if multimodal_result['mae'] < min(text_result['mae'], visual_result['mae']):
    print("\n4. Multimodal fusion improves MAE (best predictions)")
else:
    print("\n4. Multimodal fusion DOES NOT improve MAE")
    best_single = 'Text' if text_result['mae'] < visual_result['mae'] else 'Visual'
    print(f"   {best_single}-only model has better MAE")

# Recommendations
print("\n[REC] Recommendations:")
print("-" * 80)

if text_result['mae'] < multimodal_result['mae']:
    print("1. PRODUCTION: Use Text-only model (Phase 4a)")
    print(f"   Reason: Best MAE ({text_result['mae']:.2f} vs {multimodal_result['mae']:.2f})")

if multimodal_result['r2'] > text_result['r2']:
    print("\n2. RESEARCH: Use Multimodal model (Phase 4b)")
    print(f"   Reason: Best R² ({multimodal_result['r2']:.4f} vs {text_result['r2']:.4f})")

if visual_result['r2'] < text_result['r2']:
    print("\n3. IMPROVEMENT NEEDED: Visual embeddings underperforming")
    print("   - Increase ViT PCA components (50 -> 75-100)")
    print("   - Implement VideoMAE for video posts")
    print(f"   - Current ViT variance: {vit_variance*100:.1f}% (target: 90%+)")

print("\n" + "=" * 80)
print("EXPERIMENTS COMPLETE!")
print("=" * 80 + "\n")

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('experiments/modality_comparison_results.csv', index=False)
print(f"[SAVE] Results saved to: experiments/modality_comparison_results.csv")
