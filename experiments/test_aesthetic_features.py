#!/usr/bin/env python3
"""
Test Aesthetic Quality Features vs Current Best
NIMA-inspired + Composition + Saliency vs Aspect Ratio + Contrast
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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "="*90)
print(" "*25 + "AESTHETIC FEATURES EXPERIMENT")
print(" "*20 + "NIMA vs Current Best (Contrast + Aspect Ratio)")
print("="*90)

# Load data
print("\n[DATA] Loading datasets...")
baseline_df = pd.read_csv('data/processed/baseline_dataset.csv')
bert_df = pd.read_csv('data/processed/bert_embeddings.csv')
aesthetic_df = pd.read_csv('data/processed/aesthetic_features.csv')

# Previous best features
try:
    enhanced_df = pd.read_csv('data/processed/enhanced_visual_features.csv')
    print(f"   Enhanced features: {len(enhanced_df)} posts")
except:
    print("[WARN] Enhanced features not found, using only aesthetic features")
    enhanced_df = None

print(f"   Baseline: {len(baseline_df)} posts")
print(f"   BERT: {len(bert_df)} posts")
print(f"   Aesthetic: {len(aesthetic_df)} posts")

# Merge aesthetic features with baseline
merged_df = baseline_df.merge(aesthetic_df, on='post_id', how='left')
print(f"   Merged: {len(merged_df)} posts with aesthetic features")

# Count posts with aesthetic features
non_zero_aesthetic = merged_df[merged_df['aesthetic_sharpness'] != 0]
print(f"   Posts with aesthetic features: {len(non_zero_aesthetic)} ({len(non_zero_aesthetic)/len(merged_df)*100:.1f}%)")

# BERT PCA
bert_cols = [col for col in bert_df.columns if col.startswith('bert_dim_')]
pca_bert = PCA(n_components=50, random_state=42)
X_bert_reduced = pd.DataFrame(
    pca_bert.fit_transform(bert_df[bert_cols]),
    columns=[f'bert_pc_{i}' for i in range(50)]
)
print(f"   BERT: 768 -> 50 dims ({pca_bert.explained_variance_ratio_.sum()*100:.1f}% variance)")

# Baseline features
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']

# Target
y = merged_df['likes'].copy()

# Aesthetic feature groups
aesthetic_nima = ['aesthetic_sharpness', 'aesthetic_noise', 'aesthetic_brightness',
                  'aesthetic_exposure_quality', 'aesthetic_color_harmony',
                  'aesthetic_saturation', 'aesthetic_saturation_variance',
                  'aesthetic_luminance_contrast']

aesthetic_composition = ['composition_rule_of_thirds', 'composition_balance',
                        'composition_symmetry', 'composition_edge_density']

aesthetic_saliency = ['saliency_center_bias', 'saliency_attention_spread',
                     'saliency_subject_isolation']

aesthetic_color = ['color_vibrancy', 'color_warmth', 'color_diversity']

all_aesthetic = aesthetic_nima + aesthetic_composition + aesthetic_saliency + aesthetic_color

# Fill NaN with 0 (missing images)
for col in all_aesthetic:
    merged_df[col] = merged_df[col].fillna(0)

# Experiments
experiments = []

# 1. Text-only baseline
X_text = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced
], axis=1)
experiments.append({
    'name': 'Text Only (Baseline)',
    'features': X_text,
    'visual_features': []
})

# 2. Current best: Contrast + Aspect Ratio
if enhanced_df is not None:
    merged_enhanced = merged_df.merge(
        enhanced_df[['post_id', 'contrast', 'aspect_ratio']],
        on='post_id',
        how='left'
    )
    merged_enhanced['contrast'] = merged_enhanced['contrast'].fillna(0)
    merged_enhanced['aspect_ratio'] = merged_enhanced['aspect_ratio'].fillna(0)

    X_current_best = pd.concat([
        baseline_df[baseline_cols].reset_index(drop=True),
        X_bert_reduced,
        merged_enhanced[['contrast', 'aspect_ratio']].reset_index(drop=True)
    ], axis=1)
    experiments.append({
        'name': 'Current Best (Contrast + Aspect)',
        'features': X_current_best,
        'visual_features': ['contrast', 'aspect_ratio']
    })

# 3. NIMA features (8)
X_nima = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    merged_df[aesthetic_nima].reset_index(drop=True)
], axis=1)
experiments.append({
    'name': 'Text + NIMA Features (8)',
    'features': X_nima,
    'visual_features': aesthetic_nima
})

# 4. Composition features (4)
X_composition = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    merged_df[aesthetic_composition].reset_index(drop=True)
], axis=1)
experiments.append({
    'name': 'Text + Composition (4)',
    'features': X_composition,
    'visual_features': aesthetic_composition
})

# 5. Saliency features (3)
X_saliency = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    merged_df[aesthetic_saliency].reset_index(drop=True)
], axis=1)
experiments.append({
    'name': 'Text + Saliency (3)',
    'features': X_saliency,
    'visual_features': aesthetic_saliency
})

# 6. Color appeal features (3)
X_color = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    merged_df[aesthetic_color].reset_index(drop=True)
], axis=1)
experiments.append({
    'name': 'Text + Color Appeal (3)',
    'features': X_color,
    'visual_features': aesthetic_color
})

# 7. All aesthetic features (18)
X_all_aesthetic = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    merged_df[all_aesthetic].reset_index(drop=True)
], axis=1)
experiments.append({
    'name': 'Text + All Aesthetic (18)',
    'features': X_all_aesthetic,
    'visual_features': all_aesthetic
})

# 8. Best aesthetic + current best
if enhanced_df is not None:
    X_combined = pd.concat([
        baseline_df[baseline_cols].reset_index(drop=True),
        X_bert_reduced,
        merged_enhanced[['contrast', 'aspect_ratio']].reset_index(drop=True),
        merged_df[all_aesthetic].reset_index(drop=True)
    ], axis=1)
    experiments.append({
        'name': 'Combined (Current + Aesthetic)',
        'features': X_combined,
        'visual_features': ['contrast', 'aspect_ratio'] + all_aesthetic
    })

print(f"\n[CONFIG] Testing {len(experiments)} configurations:")
for i, exp in enumerate(experiments, 1):
    print(f"   {i}. {exp['name']} ({exp['features'].shape[1]} features)")

# Run experiments
print("\n" + "="*90)
print("RUNNING EXPERIMENTS")
print("="*90)

results = []

for exp in experiments:
    print(f"\n[TEST] {exp['name']}")
    X = exp['features']

    # Train-test split
    X_train_raw, X_test_raw, y_train_orig, y_test_orig = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"   Train: {len(X_train_raw)}, Test: {len(X_test_raw)}")

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

    # Ensemble model
    print("   Training RF...", end=" ")
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

    print("HGB...", end=" ")
    hgb = HistGradientBoostingRegressor(
        max_iter=400,
        max_depth=14,
        learning_rate=0.05,
        min_samples_leaf=4,
        l2_regularization=0.1,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20
    )
    hgb.fit(X_train, y_train_log)

    # Predictions
    pred_rf_log = rf.predict(X_test)
    pred_hgb_log = hgb.predict(X_test)

    # Ensemble weights (equal for now)
    pred_log = 0.5 * pred_rf_log + 0.5 * pred_hgb_log
    pred = np.expm1(pred_log)

    # Evaluate
    mae = mean_absolute_error(y_test_orig, pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, pred))
    r2 = r2_score(y_test_orig, pred)

    # Feature importance (RF)
    feature_importance = pd.DataFrame({
        'feature': list(X.columns),
        'importance': rf.feature_importances_
    })

    visual_importance = 0.0
    for vf in exp['visual_features']:
        vf_imp = feature_importance[feature_importance['feature'] == vf]['importance'].values
        if len(vf_imp) > 0:
            visual_importance += vf_imp[0]

    results.append({
        'name': exp['name'],
        'features': X.shape[1],
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'visual_importance': visual_importance
    })

    print(f"MAE={mae:.2f}, R2={r2:.4f}, Visual={visual_importance*100:.2f}%")

# Results summary
print("\n" + "="*90)
print("RESULTS SUMMARY")
print("="*90)

baseline_mae = results[0]['mae']
baseline_r2 = results[0]['r2']

print("\n[TABLE] Performance Comparison:")
print("-" * 100)
print(f"{'Configuration':<40} | {'Feat':>5} | {'MAE':>8} | {'R2':>8} | {'vs Base':>10} | {'Visual%':>9}")
print("-" * 100)

for res in results:
    mae_change = ((baseline_mae - res['mae']) / baseline_mae) * 100
    symbol = "+" if mae_change > 0 else ""

    print(f"{res['name']:<40} | {res['features']:>5} | {res['mae']:>8.2f} | "
          f"{res['r2']:>8.4f} | {symbol}{mae_change:>9.2f}% | "
          f"{res['visual_importance']*100:>8.2f}%")

print("-" * 100)

# Find best
best_model = min(results, key=lambda x: x['mae'])
print(f"\n[BEST] Best model: {best_model['name']}")
print(f"   MAE: {best_model['mae']:.2f}")
print(f"   R2: {best_model['r2']:.4f}")
print(f"   Improvement vs baseline: {((baseline_mae-best_model['mae'])/baseline_mae)*100:+.2f}%")

# Statistical test if current best exists
if len(results) >= 2:
    current_best_mae = results[1]['mae'] if enhanced_df is not None else baseline_mae
    best_aesthetic_mae = best_model['mae']

    # Use bootstrap for significance test
    print(f"\n[STATS] Comparing best aesthetic vs current best:")
    print(f"   Current best MAE: {current_best_mae:.2f}")
    print(f"   Best aesthetic MAE: {best_aesthetic_mae:.2f}")
    improvement = ((current_best_mae - best_aesthetic_mae) / current_best_mae) * 100
    print(f"   Improvement: {improvement:+.2f}%")

    if improvement > 0:
        print(f"   [OK] Aesthetic features BETTER by {improvement:.2f}%")
    else:
        print(f"   [FAIL] Aesthetic features WORSE by {abs(improvement):.2f}%")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('experiments/aesthetic_features_results.csv', index=False)
print(f"\n[SAVE] Results saved to: experiments/aesthetic_features_results.csv")

print("\n" + "="*90)
print("AESTHETIC FEATURES EXPERIMENT COMPLETE!")
print("="*90)
print("")
