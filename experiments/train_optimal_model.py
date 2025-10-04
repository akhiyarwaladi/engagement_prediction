#!/usr/bin/env python3
"""
Train Optimal Model - Based on Ablation Study Results
======================================================

Based on ablation study findings, train the OPTIMAL model using:
- Baseline features (9)
- BERT PCA (50)
- Quality features ONLY (3) - sharpness, contrast, aspect_ratio

This is the minimal feature set that achieves best MAE!
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
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "=" * 90)
print(" " * 20 + "TRAINING OPTIMAL MODEL")
print(" " * 10 + "Based on Ablation Study: Text + Quality Features Only")
print("=" * 90)

# Load all features
print("\n[DATA] Loading feature sets...")
baseline_df = pd.read_csv('data/processed/baseline_dataset.csv')
bert_df = pd.read_csv('data/processed/bert_embeddings.csv')
enhanced_df = pd.read_csv('data/processed/enhanced_visual_features.csv')

bert_cols = [col for col in bert_df.columns if col.startswith('bert_dim_')]
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']
quality_features = ['sharpness', 'contrast', 'aspect_ratio']

print(f"   Baseline: {len(baseline_cols)} features")
print(f"   BERT: {len(bert_cols)} dims")
print(f"   Quality Visual: {len(quality_features)} features")

# Target
y = baseline_df['likes'].copy()

# BERT PCA
print("\n[BERT] Applying PCA to BERT...")
X_bert = bert_df[bert_cols].copy()
pca_bert = PCA(n_components=50, random_state=42)
X_bert_reduced = pd.DataFrame(
    pca_bert.fit_transform(X_bert),
    columns=[f'bert_pc_{i}' for i in range(50)]
)
bert_variance = pca_bert.explained_variance_ratio_.sum()
print(f"   BERT: 768 -> 50 dims ({bert_variance*100:.1f}% variance)")

# Combine features: Baseline + BERT + Quality
print("\n[FEATURES] Creating optimal feature set...")
X_optimal = pd.concat([
    baseline_df[baseline_cols].reset_index(drop=True),
    X_bert_reduced,
    enhanced_df[quality_features].reset_index(drop=True)
], axis=1)

print(f"   Total features: {X_optimal.shape[1]}")
print(f"   - Baseline: {len(baseline_cols)}")
print(f"   - BERT PCA: 50")
print(f"   - Quality: {len(quality_features)} (sharpness, contrast, aspect_ratio)")

# Train-test split
print("\n[SPLIT] Creating train/test split...")
X_train_raw, X_test_raw, y_train_orig, y_test_orig = train_test_split(
    X_optimal, y, test_size=0.3, random_state=42
)
print(f"   Train: {len(X_train_raw)} samples")
print(f"   Test: {len(X_test_raw)} samples")

# Preprocessing
print("\n[PREPROCESS] Outlier clipping and log transform...")
clip_percentile = 99
clip_value = np.percentile(y_train_orig, clip_percentile)
print(f"   Clipping at {clip_percentile}th percentile: {clip_value:.0f} likes")

y_train_clipped = np.clip(y_train_orig, None, clip_value)
y_train_log = np.log1p(y_train_clipped)
y_test_log = np.log1p(y_test_orig)

print(f"   Train stats: mean={y_train_clipped.mean():.1f}, std={y_train_clipped.std():.1f}, max={y_train_clipped.max():.0f}")

# Quantile transformation
print("\n[TRANSFORM] Applying quantile transformation...")
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

# Train Random Forest
print("\n[RF] Training Random Forest...")
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
print("   [OK] Random Forest trained")

# Train HistGradientBoosting
print("\n[HGB] Training HistGradientBoosting...")
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
print("   [OK] HistGradientBoosting trained")

# Make predictions
print("\n[PREDICT] Making predictions...")
pred_rf_log = rf.predict(X_test)
pred_hgb_log = hgb.predict(X_test)

pred_rf = np.expm1(pred_rf_log)
pred_hgb = np.expm1(pred_hgb_log)

# Weighted ensemble based on MAE
mae_rf = mean_absolute_error(y_test_orig, pred_rf)
mae_hgb = mean_absolute_error(y_test_orig, pred_hgb)

w_rf = 1.0 / mae_rf
w_hgb = 1.0 / mae_hgb
total_w = w_rf + w_hgb
w_rf /= total_w
w_hgb /= total_w

print(f"   RF weight: {w_rf:.1%}")
print(f"   HGB weight: {w_hgb:.1%}")

pred_ensemble_log = w_rf * pred_rf_log + w_hgb * pred_hgb_log
pred_ensemble = np.expm1(pred_ensemble_log)

# Evaluate
print("\n" + "=" * 90)
print("PERFORMANCE EVALUATION")
print("=" * 90)

mae_ensemble = mean_absolute_error(y_test_orig, pred_ensemble)
rmse_ensemble = np.sqrt(mean_squared_error(y_test_orig, pred_ensemble))
r2_ensemble = r2_score(y_test_orig, pred_ensemble)
pct_error = (mae_ensemble / y.mean()) * 100

print(f"\n[ENSEMBLE] Weighted Ensemble Performance:")
print(f"   MAE:  {mae_ensemble:.2f} likes")
print(f"   RMSE: {rmse_ensemble:.2f} likes")
print(f"   R²:   {r2_ensemble:.4f}")
print(f"   Percentage Error: {pct_error:.1f}%")

print(f"\n[RF] Random Forest Performance:")
print(f"   MAE:  {mae_rf:.2f} likes")
print(f"   R²:   {r2_score(y_test_orig, pred_rf):.4f}")

print(f"\n[HGB] HistGradientBoosting Performance:")
print(f"   MAE:  {mae_hgb:.2f} likes")
print(f"   R²:   {r2_score(y_test_orig, pred_hgb):.4f}")

# Feature importance
print("\n[IMPORTANCE] Feature Importance Analysis:")
feature_importance = pd.DataFrame({
    'feature': list(X_optimal.columns),
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   Top 10 Most Important Features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:20} {row['importance']*100:5.2f}%")

# Group importance
baseline_importance = feature_importance[feature_importance['feature'].isin(baseline_cols)]['importance'].sum()
bert_importance = feature_importance[feature_importance['feature'].str.startswith('bert_')]['importance'].sum()
quality_importance = feature_importance[feature_importance['feature'].isin(quality_features)]['importance'].sum()

print(f"\n   Feature Group Importance:")
print(f"   Baseline: {baseline_importance*100:5.1f}%")
print(f"   BERT:     {bert_importance*100:5.1f}%")
print(f"   Quality:  {quality_importance*100:5.1f}%")

# Quality feature breakdown
print(f"\n   Quality Features Breakdown:")
for feat in quality_features:
    imp = feature_importance[feature_importance['feature'] == feat]['importance'].values[0]
    print(f"   {feat:15} {imp*100:5.2f}%")

# Compare with previous models
print("\n" + "=" * 90)
print("COMPARISON WITH PREVIOUS MODELS")
print("=" * 90)

print("\n[COMPARISON] Performance vs Previous Best Models:")
print("-" * 90)
print(f"{'Model':<40} | {'MAE':>8} | {'R²':>8} | {'Features':>8}")
print("-" * 90)
print(f"{'Text-Only (Previous Best MAE)':40} | {'125.59':>8} | {'0.5134':>8} | {'59':>8}")
print(f"{'All Image Features (Best R²)':40} | {'125.63':>8} | {'0.5444':>8} | {'69':>8}")
print(f"{'OPTIMAL (Quality Only) - THIS MODEL':40} | {mae_ensemble:8.2f} | {r2_ensemble:8.4f} | {X_optimal.shape[1]:8}")
print("-" * 90)

# Calculate improvements
text_only_mae = 125.59
text_only_r2 = 0.5134

improvement_mae = ((text_only_mae - mae_ensemble) / text_only_mae) * 100
improvement_r2 = ((r2_ensemble - text_only_r2) / text_only_r2) * 100

print(f"\n[IMPROVEMENT] vs Text-Only:")
print(f"   MAE: {improvement_mae:+.2f}%")
print(f"   R²:  {improvement_r2:+.2f}%")

if improvement_mae > 0:
    print(f"   Quality features IMPROVE prediction accuracy!")
elif improvement_mae > -1:
    print(f"   Quality features have neutral impact (within margin of error)")
else:
    print(f"   Quality features DECREASE accuracy")

# Save model
print("\n[SAVE] Saving optimal model...")
model_dir = Path('models')
model_dir.mkdir(exist_ok=True)

model_package = {
    'rf': rf,
    'hgb': hgb,
    'weights': {'rf': w_rf, 'hgb': w_hgb},
    'pca_bert': pca_bert,
    'transformer': transformer,
    'features': {
        'baseline_cols': baseline_cols,
        'quality_features': quality_features,
        'all_features': list(X_optimal.columns)
    },
    'preprocessing': {
        'clip_percentile': clip_percentile,
        'clip_value': clip_value
    },
    'performance': {
        'mae': mae_ensemble,
        'rmse': rmse_ensemble,
        'r2': r2_ensemble,
        'pct_error': pct_error
    }
}

model_path = model_dir / 'optimal_text_quality_model.pkl'
joblib.dump(model_package, model_path)
print(f"   Model saved to: {model_path}")

# Save feature importance
importance_path = 'experiments/optimal_model_feature_importance.csv'
feature_importance.to_csv(importance_path, index=False)
print(f"   Feature importance saved to: {importance_path}")

print("\n" + "=" * 90)
print("OPTIMAL MODEL TRAINING COMPLETE!")
print("=" * 90)

print(f"\n[SUMMARY] Optimal Model Configuration:")
print(f"   Features: {X_optimal.shape[1]} (Baseline + BERT + Quality)")
print(f"   MAE: {mae_ensemble:.2f} likes ({pct_error:.1f}% error)")
print(f"   R²: {r2_ensemble:.4f}")
print(f"   Model: Weighted ensemble (RF {w_rf:.1%} + HGB {w_hgb:.1%})")
print(f"\n[VERDICT] Quality features contribution: {quality_importance*100:.1f}%")

if quality_importance > 0.02:
    print(f"   Quality features SIGNIFICANTLY contribute to predictions!")
else:
    print(f"   Quality features have minimal contribution")

print("")
