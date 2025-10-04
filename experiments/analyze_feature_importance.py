#!/usr/bin/env python3
"""
Feature Importance Analysis - Ablation Study
=============================================

Test each enhanced visual feature GROUP individually to answer:
1. Does face detection (face_count) significantly improve predictions?
2. Which visual feature groups contribute most?
3. What's the optimal minimal feature set?

Feature Groups:
- Face Features: face_count
- Text Features: has_text, text_density
- Color Features: brightness, dominant_hue, saturation, color_variance
- Quality Features: sharpness, contrast, aspect_ratio
- Video Features: video_duration, video_fps, video_frames, video_brightness, video_motion
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
print(" " * 20 + "FEATURE IMPORTANCE ABLATION STUDY")
print(" " * 15 + "Which Enhanced Visual Features Matter Most?")
print("=" * 90)

# Load all features
print("\n[DATA] Loading feature sets...")
baseline_df = pd.read_csv('data/processed/baseline_dataset.csv')
bert_df = pd.read_csv('data/processed/bert_embeddings.csv')
enhanced_df = pd.read_csv('data/processed/enhanced_visual_features.csv')

bert_cols = [col for col in bert_df.columns if col.startswith('bert_dim_')]
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']

# Define enhanced feature groups
face_features = ['face_count']
text_features = ['has_text', 'text_density']
color_features = ['brightness', 'dominant_hue', 'saturation', 'color_variance']
quality_features = ['sharpness', 'contrast', 'aspect_ratio']
video_features = ['video_duration', 'video_fps', 'video_frames', 'video_brightness', 'video_motion']

all_enhanced_cols = face_features + text_features + color_features + quality_features + video_features

print(f"   Baseline: {len(baseline_cols)} features")
print(f"   BERT: {len(bert_cols)} dims")
print(f"   Enhanced Visual Groups:")
print(f"     - Face: {len(face_features)} features")
print(f"     - Text: {len(text_features)} features")
print(f"     - Color: {len(color_features)} features")
print(f"     - Quality: {len(quality_features)} features")
print(f"     - Video: {len(video_features)} features")

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

# Test configurations - ABLATION STUDY
experiments = []

# Baseline: Text-only (no visual)
X_text_only = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced
], axis=1)
experiments.append({
    'name': 'Text-Only Baseline',
    'features': X_text_only,
    'description': 'No visual features (59 features)',
    'group': 'baseline'
})

# Test 1: ONLY Face Detection
X_face_only = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    enhanced_df[face_features].reset_index(drop=True)
], axis=1)
experiments.append({
    'name': 'Face Detection Only',
    'features': X_face_only,
    'description': f'BERT + face_count ({60} features)',
    'group': 'face'
})

# Test 2: ONLY Text Detection
X_text_detect = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    enhanced_df[text_features].reset_index(drop=True)
], axis=1)
experiments.append({
    'name': 'Text Detection Only',
    'features': X_text_detect,
    'description': f'BERT + has_text + text_density ({61} features)',
    'group': 'text'
})

# Test 3: ONLY Color Features
X_color_only = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    enhanced_df[color_features].reset_index(drop=True)
], axis=1)
experiments.append({
    'name': 'Color Features Only',
    'features': X_color_only,
    'description': f'BERT + color features ({63} features)',
    'group': 'color'
})

# Test 4: ONLY Quality Features
X_quality_only = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    enhanced_df[quality_features].reset_index(drop=True)
], axis=1)
experiments.append({
    'name': 'Quality Features Only',
    'features': X_quality_only,
    'description': f'BERT + quality features ({62} features)',
    'group': 'quality'
})

# Test 5: ONLY Video Features
X_video_only = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    enhanced_df[video_features].reset_index(drop=True)
], axis=1)
experiments.append({
    'name': 'Video Features Only',
    'features': X_video_only,
    'description': f'BERT + video features ({64} features)',
    'group': 'video'
})

# Test 6: Face + Text (Social Proof + Infographics)
X_face_text = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    enhanced_df[face_features + text_features].reset_index(drop=True)
], axis=1)
experiments.append({
    'name': 'Face + Text Detection',
    'features': X_face_text,
    'description': f'BERT + face + text detection ({62} features)',
    'group': 'combined'
})

# Test 7: ALL Image Features (no video)
image_features = face_features + text_features + color_features + quality_features
X_image_all = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    enhanced_df[image_features].reset_index(drop=True)
], axis=1)
experiments.append({
    'name': 'All Image Features',
    'features': X_image_all,
    'description': f'BERT + all image features ({69} features)',
    'group': 'combined'
})

# Test 8: ALL Enhanced Visual Features
X_all_enhanced = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    enhanced_df[all_enhanced_cols].reset_index(drop=True)
], axis=1)
experiments.append({
    'name': 'All Enhanced Visual',
    'features': X_all_enhanced,
    'description': f'BERT + all enhanced features ({74} features)',
    'group': 'combined'
})

# Run experiments
print("\n" + "=" * 90)
print("RUNNING ABLATION STUDY")
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

    # Calculate improvement over baseline
    baseline_mae = None
    improvement_mae = 0.0
    improvement_r2 = 0.0

    results.append({
        'name': exp['name'],
        'group': exp['group'],
        'features': X.shape[1],
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pct_error': pct_error
    })

    print(f"   MAE: {mae:.2f} likes ({pct_error:.1f}% error)")
    print(f"   R2: {r2:.4f}")

# Results summary
print("\n" + "=" * 90)
print("ABLATION STUDY RESULTS")
print("=" * 90)

# Get baseline for comparison
baseline_result = results[0]
baseline_mae = baseline_result['mae']
baseline_r2 = baseline_result['r2']

print("\n[TABLE] Performance Comparison:")
print("-" * 110)
print(f"{'Configuration':<30} | {'Features':>8} | {'MAE':>8} | {'R2':>8} | {'vs Baseline MAE':>15} | {'vs Baseline R2':>14}")
print("-" * 110)

for res in results:
    mae_change = ((baseline_mae - res['mae']) / baseline_mae) * 100
    r2_change = ((res['r2'] - baseline_r2) / baseline_r2) * 100

    mae_symbol = "+" if mae_change > 0 else ""
    r2_symbol = "+" if r2_change > 0 else ""

    print(f"{res['name']:<30} | {res['features']:>8} | {res['mae']:>8.2f} | "
          f"{res['r2']:>8.4f} | {mae_symbol}{mae_change:>13.1f}% | {r2_symbol}{r2_change:>12.1f}%")

print("-" * 110)

# Find best
best_mae = min(results, key=lambda x: x['mae'])
best_r2 = max(results, key=lambda x: x['r2'])

print(f"\n[BEST] Best MAE: {best_mae['name']} ({best_mae['mae']:.2f} likes)")
print(f"[BEST] Best R2: {best_r2['name']} ({best_r2['r2']:.4f})")

# Answer specific questions
print("\n" + "=" * 90)
print("RESEARCH QUESTIONS ANSWERED")
print("=" * 90)

# Question 1: Does face detection help?
face_result = [r for r in results if r['name'] == 'Face Detection Only'][0]
face_mae_improve = ((baseline_mae - face_result['mae']) / baseline_mae) * 100
face_r2_improve = ((face_result['r2'] - baseline_r2) / baseline_r2) * 100

print("\n[Q1] Does face detection (face_count) significantly improve predictions?")
print("-" * 90)
if face_mae_improve > 0:
    print(f"   YES! Face detection HELPS")
    print(f"   MAE improvement: +{face_mae_improve:.1f}% ({baseline_mae:.2f} -> {face_result['mae']:.2f})")
    print(f"   R2 improvement: +{face_r2_improve:.1f}% ({baseline_r2:.4f} -> {face_result['r2']:.4f})")
else:
    print(f"   NO. Face detection alone doesn't help")
    print(f"   MAE change: {face_mae_improve:.1f}% ({baseline_mae:.2f} -> {face_result['mae']:.2f})")
    print(f"   R2 change: {face_r2_improve:.1f}% ({baseline_r2:.4f} -> {face_result['r2']:.4f})")

# Question 2: Which feature group contributes most?
print("\n[Q2] Which enhanced visual feature group contributes most?")
print("-" * 90)

group_results = [
    ('Face', [r for r in results if r['name'] == 'Face Detection Only'][0]),
    ('Text', [r for r in results if r['name'] == 'Text Detection Only'][0]),
    ('Color', [r for r in results if r['name'] == 'Color Features Only'][0]),
    ('Quality', [r for r in results if r['name'] == 'Quality Features Only'][0]),
    ('Video', [r for r in results if r['name'] == 'Video Features Only'][0])
]

# Sort by MAE improvement
group_results_sorted = sorted(group_results, key=lambda x: x[1]['mae'])

print("   Ranking by MAE (best to worst):")
for i, (group_name, res) in enumerate(group_results_sorted, 1):
    mae_improve = ((baseline_mae - res['mae']) / baseline_mae) * 100
    symbol = "+" if mae_improve > 0 else ""
    print(f"   {i}. {group_name:10} MAE={res['mae']:.2f} ({symbol}{mae_improve:.1f}% vs baseline)")

# Question 3: What's the optimal minimal feature set?
print("\n[Q3] What's the optimal minimal feature set?")
print("-" * 90)

# Find minimal feature set with best performance
minimal_results = [r for r in results if r['features'] <= 65]  # Up to ~5 visual features
minimal_best = min(minimal_results, key=lambda x: x['mae'])

print(f"   Best minimal set: {minimal_best['name']}")
print(f"   Features: {minimal_best['features']}")
print(f"   MAE: {minimal_best['mae']:.2f} likes")
print(f"   R2: {minimal_best['r2']:.4f}")

mae_improve_min = ((baseline_mae - minimal_best['mae']) / baseline_mae) * 100
print(f"   Improvement over text-only: {mae_improve_min:+.1f}% MAE")

# Question 4: Is combining features better?
print("\n[Q4] Is combining feature groups better than individual groups?")
print("-" * 90)

face_text_result = [r for r in results if r['name'] == 'Face + Text Detection'][0]
all_image_result = [r for r in results if r['name'] == 'All Image Features'][0]
all_enhanced_result = [r for r in results if r['name'] == 'All Enhanced Visual'][0]

print(f"   Face + Text: MAE={face_text_result['mae']:.2f}, R2={face_text_result['r2']:.4f}")
print(f"   All Image: MAE={all_image_result['mae']:.2f}, R2={all_image_result['r2']:.4f}")
print(f"   All Enhanced: MAE={all_enhanced_result['mae']:.2f}, R2={all_enhanced_result['r2']:.4f}")

if all_enhanced_result['mae'] < face_text_result['mae']:
    print("\n   YES! Combining all features is better than subset")
else:
    print("\n   NO. Simpler feature set performs better (avoid feature dilution)")

# Summary recommendations
print("\n" + "=" * 90)
print("RECOMMENDATIONS")
print("=" * 90)

print("\n1. FEATURE SELECTION:")
if best_mae['group'] == 'baseline':
    print("   - Text-only remains best, visual features don't help")
elif best_mae['group'] == 'combined':
    print(f"   - Use {best_mae['name']} ({best_mae['features']} features)")
    print(f"   - Combining multiple visual feature types works best")
else:
    print(f"   - Use {best_mae['name']} ({best_mae['features']} features)")
    print(f"   - Single feature group sufficient")

print("\n2. FACE DETECTION:")
if face_mae_improve > 1.0:
    print(f"   - Face detection MATTERS (+{face_mae_improve:.1f}% improvement)")
    print("   - Keep face_count feature in final model")
else:
    print(f"   - Face detection has minimal impact ({face_mae_improve:+.1f}%)")
    print("   - Consider removing to simplify model")

print("\n3. PRIORITY FEATURES:")
top_3_groups = group_results_sorted[:3]
print("   Focus on these feature groups (in order):")
for i, (group_name, res) in enumerate(top_3_groups, 1):
    mae_improve = ((baseline_mae - res['mae']) / baseline_mae) * 100
    print(f"   {i}. {group_name} features ({mae_improve:+.1f}% MAE improvement)")

print("\n" + "=" * 90)
print("ABLATION STUDY COMPLETE!")
print("=" * 90 + "\n")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('experiments/feature_ablation_results.csv', index=False)
print(f"[SAVE] Results saved to: experiments/feature_ablation_results.csv")
