#!/usr/bin/env python3
"""
Train Models on Multi-Account Dataset (8,610 Posts)
====================================================

Train and compare 3 models:
1. Baseline: 9 baseline features only
2. BERT Model: 9 baseline + 50 BERT PCA features
3. Full Model: 9 baseline + 50 BERT + 8 NIMA aesthetic features

Previous best (1,949 posts): MAE=94.54 (Baseline + BERT)
Target (8,610 posts): MAE < 60
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*90)
print(" "*20 + "MULTI-ACCOUNT MODEL TRAINING")
print(" "*15 + "8,610 Posts from 8 UNJA Accounts")
print("="*90)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n[LOAD] Loading datasets...")
multi_account_df = pd.read_csv('multi_account_dataset.csv')
bert_df = pd.read_csv('data/processed/bert_embeddings_multi_account.csv')
aesthetic_df = pd.read_csv('data/processed/aesthetic_features_multi_account.csv')

print(f"   Multi-account data: {len(multi_account_df)} posts")
print(f"   BERT embeddings: {len(bert_df)} posts")
print(f"   Aesthetic features: {len(aesthetic_df)} posts")

# Merge datasets
merged_df = multi_account_df.copy()
merged_df = merged_df.merge(aesthetic_df, on=['post_id', 'account'], how='left')

print(f"\n[MERGE] Merged dataset: {len(merged_df)} posts")

# Account breakdown
print(f"\n[ACCOUNTS] Posts per account:")
for account in sorted(merged_df['account'].unique()):
    count = len(merged_df[merged_df['account'] == account])
    print(f"   {account:25} {count:5} posts")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

print("\n[FEATURES] Preparing feature matrices...")

# Baseline features
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']

X_baseline = merged_df[baseline_cols].copy()

# BERT PCA (768 -> 50 dims)
bert_cols = [col for col in bert_df.columns if col.startswith('bert_dim_')]
pca_bert = PCA(n_components=50, random_state=42)
X_bert_reduced = pd.DataFrame(
    pca_bert.fit_transform(bert_df[bert_cols]),
    columns=[f'bert_pc_{i}' for i in range(50)]
)

print(f"   BERT PCA: 768 -> 50 dims ({pca_bert.explained_variance_ratio_.sum()*100:.1f}% variance)")

# NIMA aesthetic features
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

# Fill NaN with 0 for missing aesthetic features
for col in nima_features:
    merged_df[col] = merged_df[col].fillna(0)

X_nima = merged_df[nima_features].copy()

# Target variable
y = merged_df['likes'].copy()

# Build 3 feature matrices
# Model 1: Baseline only (9 features)
X_model1 = X_baseline.reset_index(drop=True)

# Model 2: Baseline + BERT (9 + 50 = 59 features)
X_model2 = pd.concat([
    X_baseline.reset_index(drop=True),
    X_bert_reduced
], axis=1)

# Model 3: Baseline + BERT + NIMA (9 + 50 + 8 = 67 features)
X_model3 = pd.concat([
    X_baseline.reset_index(drop=True),
    X_bert_reduced,
    X_nima.reset_index(drop=True)
], axis=1)

print(f"\n[MODELS] Feature counts:")
print(f"   Model 1 (Baseline):       {X_model1.shape[1]:2} features")
print(f"   Model 2 (Baseline+BERT):  {X_model2.shape[1]:2} features")
print(f"   Model 3 (Full):           {X_model3.shape[1]:2} features")

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================

print(f"\n[SPLIT] Creating train/test splits...")

# Split all 3 datasets with same random seed
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X_model1, y, test_size=0.2, random_state=42
)

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X_model2, y, test_size=0.2, random_state=42
)

X3_train, X3_test, y3_train, y3_test = train_test_split(
    X_model3, y, test_size=0.2, random_state=42
)

print(f"   Train size: {len(X1_train):,} posts")
print(f"   Test size:  {len(X1_test):,} posts")

# ============================================================================
# 4. PREPROCESSING
# ============================================================================

print(f"\n[PREPROCESS] Applying transformations...")

def preprocess_data(X_train, y_train, X_test, y_test):
    """Apply outlier clipping, log transform, and quantile scaling"""

    # Clip outliers at 99th percentile
    clip_value = np.percentile(y_train, 99)
    y_train_clipped = np.clip(y_train, None, clip_value)

    # Log transform
    y_train_log = np.log1p(y_train_clipped)
    y_test_log = np.log1p(y_test)

    # Quantile transform features
    transformer = QuantileTransformer(
        n_quantiles=min(1000, len(X_train)),
        output_distribution='normal',
        random_state=42
    )

    X_train_transformed = pd.DataFrame(
        transformer.fit_transform(X_train),
        columns=X_train.columns
    )

    X_test_transformed = pd.DataFrame(
        transformer.transform(X_test),
        columns=X_test.columns
    )

    return X_train_transformed, X_test_transformed, y_train_log, y_test_log, y_test, transformer

# Preprocess all 3 models
X1_train_t, X1_test_t, y1_train_log, y1_test_log, y1_test_orig, transformer1 = preprocess_data(
    X1_train, y1_train, X1_test, y1_test
)

X2_train_t, X2_test_t, y2_train_log, y2_test_log, y2_test_orig, transformer2 = preprocess_data(
    X2_train, y2_train, X2_test, y2_test
)

X3_train_t, X3_test_t, y3_train_log, y3_test_log, y3_test_orig, transformer3 = preprocess_data(
    X3_train, y3_train, X3_test, y3_test
)

print(f"   Outlier clipping: 99th percentile")
print(f"   Log transform: applied")
print(f"   Quantile scaling: normal distribution")

# ============================================================================
# 5. TRAIN MODELS
# ============================================================================

def train_ensemble(X_train, y_train_log, X_test, model_name):
    """Train Random Forest + HistGradientBoosting ensemble"""

    print(f"\n[TRAIN] Training {model_name}...")

    # Random Forest
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
    print(f"   [OK] Random Forest trained")

    # HistGradientBoosting
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
    print(f"   [OK] HistGradientBoosting trained")

    # Ensemble predictions (50/50 weighting)
    pred_rf_log = rf.predict(X_test)
    pred_hgb_log = hgb.predict(X_test)
    pred_log = 0.5 * pred_rf_log + 0.5 * pred_hgb_log
    pred = np.expm1(pred_log)

    return {
        'rf': rf,
        'hgb': hgb,
        'predictions': pred
    }

# Train Model 1: Baseline
result1 = train_ensemble(X1_train_t, y1_train_log, X1_test_t, "Model 1 (Baseline)")

# Train Model 2: Baseline + BERT
result2 = train_ensemble(X2_train_t, y2_train_log, X2_test_t, "Model 2 (Baseline+BERT)")

# Train Model 3: Full
result3 = train_ensemble(X3_train_t, y3_train_log, X3_test_t, "Model 3 (Full)")

# ============================================================================
# 6. EVALUATE MODELS
# ============================================================================

print(f"\n" + "="*90)
print(" "*30 + "MODEL EVALUATION")
print("="*90)

def evaluate_model(y_test, y_pred, model_name):
    """Calculate and display evaluation metrics"""
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Mean baseline (predicting mean likes)
    mean_baseline = np.full_like(y_pred, y_test.mean())
    mae_baseline = mean_absolute_error(y_test, mean_baseline)
    improvement = ((mae_baseline - mae) / mae_baseline) * 100

    print(f"\n{model_name}")
    print(f"   MAE:  {mae:.2f} likes")
    print(f"   RMSE: {rmse:.2f} likes")
    print(f"   R²:   {r2:.4f}")
    print(f"   Improvement over mean baseline: {improvement:.1f}%")

    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'improvement': improvement}

metrics1 = evaluate_model(y1_test_orig, result1['predictions'], "Model 1 (Baseline - 9 features)")
metrics2 = evaluate_model(y2_test_orig, result2['predictions'], "Model 2 (Baseline+BERT - 59 features)")
metrics3 = evaluate_model(y3_test_orig, result3['predictions'], "Model 3 (Full - 67 features)")

# ============================================================================
# 7. COMPARISON
# ============================================================================

print(f"\n" + "="*90)
print(" "*30 + "MODEL COMPARISON")
print("="*90)

print(f"\n{'Model':<30} {'Features':<12} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
print("-"*90)
print(f"{'Baseline':<30} {X_model1.shape[1]:<12} {metrics1['mae']:<12.2f} {metrics1['rmse']:<12.2f} {metrics1['r2']:<12.4f}")
print(f"{'Baseline + BERT':<30} {X_model2.shape[1]:<12} {metrics2['mae']:<12.2f} {metrics2['rmse']:<12.2f} {metrics2['r2']:<12.4f}")
print(f"{'Full (Baseline+BERT+NIMA)':<30} {X_model3.shape[1]:<12} {metrics3['mae']:<12.2f} {metrics3['rmse']:<12.2f} {metrics3['r2']:<12.4f}")

# Identify champion
champion_idx = np.argmin([metrics1['mae'], metrics2['mae'], metrics3['mae']])
champion_names = ["Baseline", "Baseline+BERT", "Full"]
champion_name = champion_names[champion_idx]
champion_mae = [metrics1['mae'], metrics2['mae'], metrics3['mae']][champion_idx]
champion_metrics = [metrics1, metrics2, metrics3][champion_idx]
champion_model = [result1, result2, result3][champion_idx]
champion_features = [X_model1.shape[1], X_model2.shape[1], X_model3.shape[1]][champion_idx]
champion_X_test = [X1_test_t, X2_test_t, X3_test_t][champion_idx]
champion_transformer = [transformer1, transformer2, transformer3][champion_idx]

print(f"\n[CHAMPION] {champion_name} (MAE={champion_mae:.2f})")

# Compare to previous best
previous_best_mae = 94.54
previous_posts = 1949
improvement_vs_previous = ((previous_best_mae - champion_mae) / previous_best_mae) * 100

print(f"\n[PROGRESS] Performance vs Previous Best:")
print(f"   Previous (1,949 posts): MAE={previous_best_mae:.2f}")
print(f"   Current  (8,610 posts): MAE={champion_mae:.2f}")
print(f"   Improvement: {improvement_vs_previous:.1f}%")
print(f"   Dataset scale: {len(multi_account_df) / previous_posts:.1f}x larger")

# Target achievement
target_mae = 60.0
if champion_mae < target_mae:
    print(f"\n[SUCCESS] TARGET ACHIEVED! MAE < {target_mae} (actual: {champion_mae:.2f})")
else:
    gap = champion_mae - target_mae
    print(f"\n[WARNING] Target not reached. Need {gap:.2f} MAE reduction to hit {target_mae}")

# ============================================================================
# 8. SAVE MODELS
# ============================================================================

print(f"\n" + "="*90)
print(" "*30 + "SAVING MODELS")
print("="*90)

models_dir = Path('models')
models_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save all 3 models
for idx, (name, result, metrics, transformer, features) in enumerate([
    ("baseline", result1, metrics1, transformer1, X_model1.shape[1]),
    ("baseline_bert", result2, metrics2, transformer2, X_model2.shape[1]),
    ("full", result3, metrics3, transformer3, X_model3.shape[1])
], 1):

    model_path = models_dir / f"multi_account_{name}_{timestamp}.pkl"

    joblib.dump({
        'rf': result['rf'],
        'hgb': result['hgb'],
        'transformer': transformer,
        'pca_bert': pca_bert if 'bert' in name else None,
        'metrics': metrics,
        'dataset_info': {
            'posts': len(multi_account_df),
            'features': features,
            'accounts': len(multi_account_df['account'].unique()),
            'timestamp': timestamp
        }
    }, model_path)

    print(f"   Model {idx} ({name:15}): {model_path.name}")

# Save champion separately
champion_path = models_dir / f"champion_8610posts_{timestamp}.pkl"
champion_idx_map = {0: "baseline", 1: "baseline_bert", 2: "full"}
champion_type = champion_idx_map[champion_idx]

joblib.dump({
    'rf': champion_model['rf'],
    'hgb': champion_model['hgb'],
    'transformer': champion_transformer,
    'pca_bert': pca_bert if champion_type != "baseline" else None,
    'metrics': champion_metrics,
    'model_type': champion_type,
    'dataset_info': {
        'posts': len(multi_account_df),
        'features': champion_features,
        'accounts': len(multi_account_df['account'].unique()),
        'timestamp': timestamp
    }
}, champion_path)

print(f"\n   [CHAMPION] {champion_path.name}")
print(f"      Type: {champion_name}")
print(f"      MAE: {champion_mae:.2f}")

# ============================================================================
# 9. FEATURE IMPORTANCE (Champion Model)
# ============================================================================

print(f"\n" + "="*90)
print(" "*25 + "CHAMPION FEATURE IMPORTANCE")
print("="*90)

# Get feature importances from Random Forest
feature_names = champion_X_test.columns.tolist()
importances = champion_model['rf'].feature_importances_

# Sort by importance
indices = np.argsort(importances)[::-1]

print(f"\n[TOP 20] Most Important Features:")
for i, idx in enumerate(indices[:20], 1):
    print(f"   {i:2}. {feature_names[idx]:30} {importances[idx]:.6f}")

# ============================================================================
# 10. SUMMARY
# ============================================================================

print(f"\n" + "="*90)
print(" "*35 + "SUMMARY")
print("="*90)

print(f"\n[DATASET]")
print(f"   Total posts: {len(multi_account_df):,}")
print(f"   Accounts: {len(multi_account_df['account'].unique())}")
print(f"   Date range: {merged_df['datetime'].min()} to {merged_df['datetime'].max()}")

print(f"\n[BEST MODEL]")
print(f"   Type: {champion_name}")
print(f"   Features: {champion_features}")
print(f"   MAE: {champion_mae:.2f} likes")
print(f"   RMSE: {champion_metrics['rmse']:.2f} likes")
print(f"   R²: {champion_metrics['r2']:.4f}")

print(f"\n[PROGRESS]")
print(f"   Previous (1,949 posts): MAE={previous_best_mae:.2f}")
print(f"   Current  (8,610 posts): MAE={champion_mae:.2f}")
print(f"   Improvement: {improvement_vs_previous:.1f}%")

print(f"\n[FILES]")
print(f"   Champion model: {champion_path}")
print(f"   All models saved to: models/ directory")

print("\n" + "="*90)
print(" "*30 + "TRAINING COMPLETE!")
print("="*90 + "\n")
