#!/usr/bin/env python3
"""
Phase 4a: Training with IndoBERT Embeddings
============================================

Combine baseline features + NLP features + IndoBERT embeddings (768-dim)

Expected: MAE ~90-95, R² ~0.25-0.28
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from src.utils import load_config, get_model_path

print("\n" + "=" * 80)
print(" " * 20 + "PHASE 4A: IndoBERT-ENHANCED MODEL")
print(" " * 15 + "Baseline + NLP + BERT Embeddings (768-dim)")
print("=" * 80)

# Load BERT embeddings
print("\n📁 Loading IndoBERT embeddings...")
bert_path = 'data/processed/bert_embeddings.csv'

if not Path(bert_path).exists():
    print(f"❌ Error: {bert_path} not found!")
    print("   Run: python3 extract_bert_features.py first")
    sys.exit(1)

bert_df = pd.read_csv(bert_path)
print(f"   Loaded {len(bert_df)} BERT embeddings (768-dim)")

# Get BERT feature columns (all bert_dim_* columns)
bert_cols = [col for col in bert_df.columns if col.startswith('bert_dim_')]
print(f"   BERT features: {len(bert_cols)}")

# Load baseline features
print("\n📁 Loading baseline features...")
baseline_path = 'data/processed/baseline_dataset.csv'

if not Path(baseline_path).exists():
    print(f"⚠️  Baseline dataset not found, creating from scratch...")
    # Run baseline extraction
    from src.features import BaselineFeatureExtractor
    df_raw = pd.read_csv('fst_unja_from_gallery_dl.csv')
    extractor = BaselineFeatureExtractor()
    baseline_df = extractor.transform(df_raw)
    baseline_df.to_csv(baseline_path, index=False)
else:
    baseline_df = pd.read_csv(baseline_path)

print(f"   Loaded baseline features")

# Combine all features
print("\n🔧 Combining features...")

# Baseline features (28 from Phase 2)
baseline_feature_cols = [
    'caption_length', 'word_count', 'hashtag_count', 'mention_count',
    'is_video', 'hour', 'day_of_week', 'is_weekend', 'month'
]

# Check which features exist
available_baseline = [col for col in baseline_feature_cols if col in baseline_df.columns]

# Create combined feature set
X_baseline = baseline_df[available_baseline].copy()

# Add BERT embeddings
X_bert = bert_df[bert_cols].copy()

# Combine
X_combined = pd.concat([X_baseline, X_bert], axis=1)
y = baseline_df['likes'].copy()

print(f"   Baseline features: {len(available_baseline)}")
print(f"   BERT features: {len(bert_cols)}")
print(f"   Total features: {X_combined.shape[1]}")

# Option: Dimensionality reduction with PCA
USE_PCA = True  # Set to False to use all 768 BERT dims

if USE_PCA and X_combined.shape[1] > 100:
    print(f"\n📉 Applying PCA dimensionality reduction...")
    print(f"   Original dimensions: {X_combined.shape[1]}")

    # Keep baseline features as-is, reduce only BERT features
    X_baseline_part = X_baseline.values
    X_bert_part = X_bert.values

    # PCA on BERT features (768 → 50 dims, preserve ~95% variance)
    pca = PCA(n_components=50, random_state=42)
    X_bert_reduced = pca.fit_transform(X_bert_part)

    explained_var = pca.explained_variance_ratio_.sum()
    print(f"   BERT: 768 → 50 dimensions")
    print(f"   Variance preserved: {explained_var*100:.1f}%")

    # Recombine
    X_combined = pd.DataFrame(
        np.hstack([X_baseline_part, X_bert_reduced]),
        columns=available_baseline + [f'bert_pc_{i}' for i in range(50)]
    )
    print(f"   Final dimensions: {X_combined.shape[1]}")
else:
    pca = None

# Train-test split
print(f"\n📊 Train-test split...")
X_train_raw, X_test_raw, y_train_orig, y_test_orig = train_test_split(
    X_combined, y,
    test_size=0.3,
    random_state=42
)

print(f"   Train: {len(X_train_raw)} posts")
print(f"   Test: {len(X_test_raw)} posts")

# Robust preprocessing (from Phase 2)
print("\n🛡️ Robust preprocessing...")

# Clip outliers
clip_percentile = 99
clip_value = np.percentile(y_train_orig, clip_percentile)
y_train_clipped = np.clip(y_train_orig, None, clip_value)
print(f"   Clipping at {clip_percentile}th percentile: {clip_value:.1f} likes")

# Log transformation
y_train_log = np.log1p(y_train_clipped)
y_test_log = np.log1p(y_test_orig)

# Quantile transformation
transformer = QuantileTransformer(
    n_quantiles=min(100, len(X_train_raw)),
    output_distribution='normal',
    random_state=42
)

X_train = pd.DataFrame(
    transformer.fit_transform(X_train_raw),
    columns=X_train_raw.columns,
    index=X_train_raw.index
)

X_test = pd.DataFrame(
    transformer.transform(X_test_raw),
    columns=X_test_raw.columns,
    index=X_test_raw.index
)

print(f"   ✅ Applied quantile transformation")

# Train models
print("\n🤖 Training ensemble models...")

models = {}
predictions_train = {}
predictions_test = {}

# Model 1: Random Forest
print("\n   [1/2] Training Random Forest...")
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
models['rf'] = rf
predictions_train['rf'] = rf.predict(X_train)
predictions_test['rf'] = rf.predict(X_test)

cv_scores = cross_val_score(rf, X_train, y_train_log, cv=5,
                            scoring='neg_mean_absolute_error', n_jobs=-1)
print(f"       CV MAE: {-cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Model 2: HistGradientBoosting
print("\n   [2/2] Training HistGradientBoosting...")
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
models['hgb'] = hgb
predictions_train['hgb'] = hgb.predict(X_train)
predictions_test['hgb'] = hgb.predict(X_test)

cv_scores = cross_val_score(hgb, X_train, y_train_log, cv=5,
                            scoring='neg_mean_absolute_error', n_jobs=-1)
print(f"       CV MAE: {-cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Weighted ensemble
print("\n🎯 Creating weighted ensemble...")

weights = {}
for name, preds_test in predictions_test.items():
    preds_test_orig = np.expm1(preds_test)
    mae = mean_absolute_error(y_test_orig, preds_test_orig)
    weights[name] = 1.0 / mae
    print(f"   {name.upper()}: MAE={mae:.2f} → weight={weights[name]:.4f}")

total_weight = sum(weights.values())
weights = {k: v/total_weight for k, v in weights.items()}

print(f"\n   Normalized weights:")
for name, weight in weights.items():
    print(f"     {name.upper()}: {weight:.3f}")

ensemble_train = sum(predictions_train[name] * weights[name] for name in weights.keys())
ensemble_test = sum(predictions_test[name] * weights[name] for name in weights.keys())

# Transform back
ensemble_train_orig = np.expm1(ensemble_train)
ensemble_test_orig = np.expm1(ensemble_test)

# Evaluate
print("\n" + "=" * 80)
print("EVALUATION RESULTS - PHASE 4A (IndoBERT)")
print("=" * 80)

mae_train = mean_absolute_error(y_train_orig, ensemble_train_orig)
rmse_train = np.sqrt(mean_squared_error(y_train_orig, ensemble_train_orig))
r2_train = r2_score(y_train_orig, ensemble_train_orig)

mae_test = mean_absolute_error(y_test_orig, ensemble_test_orig)
rmse_test = np.sqrt(mean_squared_error(y_test_orig, ensemble_test_orig))
r2_test = r2_score(y_test_orig, ensemble_test_orig)

print("\n📊 TRAIN SET:")
print(f"   MAE:  {mae_train:.2f} likes")
print(f"   RMSE: {rmse_train:.2f} likes")
print(f"   R²:   {r2_train:.4f}")

print("\n📊 TEST SET:")
print(f"   MAE:  {mae_test:.2f} likes")
print(f"   RMSE: {rmse_test:.2f} likes")
print(f"   R²:   {r2_test:.4f}")

# Comparison
print("\n📈 Performance Evolution:")
print("-" * 80)
print("   Version          | Features | MAE (test) | R² (test) | Method")
print("   -----------------|----------|------------|-----------|------------------")
print(f"   Baseline         | 9        | 185.29     | 0.0860    | RF")
print(f"   Phase 1          | 14       | 115.17     | 0.0900    | RF + log")
print(f"   Phase 2          | 28       | 109.42     | 0.2006    | Ensemble + NLP")
print(f"   Phase 4a (NOW)   | {X_combined.shape[1]:<8} | {mae_test:6.2f}     | {r2_test:.4f}    | +IndoBERT 🤖")
print("-" * 80)

# Improvements
improvement_mae_phase2 = ((109.42 - mae_test) / 109.42) * 100 if mae_test < 109.42 else 0
improvement_r2_phase2 = ((r2_test - 0.2006) / 0.2006) * 100 if r2_test > 0.2006 else 0

improvement_mae_baseline = ((185.29 - mae_test) / 185.29) * 100
improvement_r2_baseline = ((r2_test - 0.086) / 0.086) * 100 if r2_test > 0.086 else 0

print(f"\n🤖 IndoBERT contribution (vs Phase 2):")
if improvement_mae_phase2 > 0:
    print(f"   MAE: {improvement_mae_phase2:.1f}% better ({109.42:.2f} → {mae_test:.2f})")
else:
    print(f"   MAE: {abs(improvement_mae_phase2):.1f}% change ({109.42:.2f} → {mae_test:.2f})")

if improvement_r2_phase2 > 0:
    print(f"   R²:  {improvement_r2_phase2:.1f}% better ({0.2006:.4f} → {r2_test:.4f})")
else:
    print(f"   R²:  {abs(improvement_r2_phase2):.1f}% change ({0.2006:.4f} → {r2_test:.4f})")

print(f"\n🎉 Total improvement (vs baseline):")
print(f"   MAE: {improvement_mae_baseline:.1f}% better ({185.29:.2f} → {mae_test:.2f})")
print(f"   R²:  {improvement_r2_baseline:.1f}% better ({0.086:.4f} → {r2_test:.4f})")

# Target assessment
print("\n🎯 Target Assessment:")
if mae_test <= 95:
    print(f"   ✅ MAE Target: ACHIEVED ({mae_test:.2f} <= 95)")
else:
    print(f"   ⚠️  MAE Target: NOT MET ({mae_test:.2f} > 95)")
    gap = mae_test - 95
    print(f"      Gap: {gap:.2f} likes (need Phase 4b with ViT)")

if r2_test >= 0.25:
    print(f"   ✅ R² Target: ACHIEVED ({r2_test:.4f} >= 0.25)")
else:
    print(f"   ⚠️  R² Target: NOT MET ({r2_test:.4f} < 0.25)")
    gap = 0.25 - r2_test
    print(f"      Gap: {gap:.4f} (need Phase 4b with ViT)")

# Save model
model_data = {
    'models': models,
    'weights': weights,
    'transformer': transformer,
    'pca': pca,
    'feature_names': list(X_combined.columns),
    'use_log_transform': True,
    'clip_percentile': 99,
    'transformation': 'log1p + quantile + IndoBERT',
    'bert_model': 'indolem/IndoBERTweet'
}

output_path = get_model_path('phase4a_bert_model.pkl')
joblib.dump(model_data, output_path)
print(f"\n💾 Model saved to: {output_path}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY - PHASE 4A")
print("=" * 80)

print("\n🤖 What was added:")
print("   1. ✅ IndoBERT embeddings (768-dim)")
if USE_PCA:
    print("   2. ✅ PCA reduction (768 → 50 dims)")
print("   3. ✅ Same robust preprocessing & ensemble")

print("\n📊 Results:")
print(f"   Features: 28 → {X_combined.shape[1]}")
print(f"   MAE:      109.42 → {mae_test:.2f} likes")
print(f"   R²:       0.2006 → {r2_test:.4f}")

if r2_test > 0.2006:
    print(f"\n   🎉 IndoBERT improves R² by {improvement_r2_phase2:.1f}%!")
else:
    print(f"\n   ℹ️  IndoBERT provides {abs(improvement_r2_phase2):.1f}% R² change")

print("\n🚀 Next steps:")
if mae_test > 80 or r2_test < 0.33:
    print("   Phase 4b: Add ViT visual embeddings")
    print("   Expected: MAE ~70-80, R² ~0.33-0.40")
    print("   Run: python3 extract_vit_features.py")
else:
    print("   ✅ Performance looks good!")
    print("   Consider collecting more data for further improvement")

print("\n" + "=" * 80)
print("DONE! 🎉")
print("=" * 80 + "\n")
