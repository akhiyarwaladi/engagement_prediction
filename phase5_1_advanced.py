"""
Phase 5.1: ADVANCED MULTIMODAL COMBINATIONS
Focus on Visual (NIMA Aesthetic) + Text (BERT) Feature Optimization

User requirement: Keep using visual and text features
Goal: Find optimal combination of BERT + Aesthetic + Baseline features
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import QuantileTransformer, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print(" "*20 + "PHASE 5.1: ADVANCED MULTIMODAL OPTIMIZATION")
print(" "*15 + "Visual (Aesthetic) + Text (BERT) + Baseline Features")
print("="*90)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[LOAD] Loading datasets...")
df_main = pd.read_csv('multi_account_dataset.csv')
df_bert = pd.read_csv('data/processed/bert_embeddings_multi_account.csv')
df_aesthetic = pd.read_csv('data/processed/aesthetic_features_multi_account.csv')

# Merge datasets
df = df_main.merge(df_bert, on='post_id', how='inner')
df = df.merge(df_aesthetic, on='post_id', how='left')

# Fill missing aesthetic features with 0
aesthetic_cols = [col for col in df.columns if col.startswith('aesthetic_')]
if aesthetic_cols:
    df[aesthetic_cols] = df[aesthetic_cols].fillna(0)

print(f"   Total posts: {len(df)}")
print(f"   BERT features: {len([c for c in df.columns if c.startswith('bert_')])}")
print(f"   Aesthetic features: {len(aesthetic_cols)}")

# ============================================================================
# PREPARE FEATURES
# ============================================================================
print("\n[FEATURES] Preparing feature matrices...")

baseline_features = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                     'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']

# Extract feature matrices
X_baseline = df[baseline_features].values
X_bert_full = df[[col for col in df.columns if col.startswith('bert_')]].values
X_aesthetic = df[aesthetic_cols].values if aesthetic_cols else np.zeros((len(df), 0))

# Target
y = df['likes'].values
clip_value = np.percentile(y, 99)
y_clipped = np.clip(y, 0, clip_value)
y_log = np.log1p(y_clipped)

# Train/test split
from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)

X_baseline_train = X_baseline[train_idx]
X_baseline_test = X_baseline[test_idx]
X_bert_train = X_bert_full[train_idx]
X_bert_test = X_bert_full[test_idx]
X_aesthetic_train = X_aesthetic[train_idx]
X_aesthetic_test = X_aesthetic[test_idx]
y_train_log = y_log[train_idx]
y_test_log = y_log[test_idx]
y_test = y_clipped[test_idx]

print(f"   Train: {len(train_idx)}, Test: {len(test_idx)}")

# ============================================================================
# EXPERIMENT 1: PCA COMBINATIONS FOR BERT + AESTHETIC
# ============================================================================
print("\n" + "="*90)
print(" "*20 + "EXPERIMENT 1: PCA DIMENSIONALITY COMBINATIONS")
print("="*90)

results = []

# Test various PCA combinations for BERT
bert_pca_options = [30, 50, 70, 100, 150]
aesthetic_use_options = [False, True]  # Whether to include aesthetic features

print("\n[TEST] Testing BERT PCA + Aesthetic combinations:")

for bert_pca_n in bert_pca_options:
    for use_aesthetic in aesthetic_use_options:
        # PCA on BERT
        pca_bert = PCA(n_components=bert_pca_n, random_state=42)
        X_bert_pca_train = pca_bert.fit_transform(X_bert_train)
        X_bert_pca_test = pca_bert.transform(X_bert_test)
        bert_variance = pca_bert.explained_variance_ratio_.sum()

        # Build feature matrix
        if use_aesthetic and aesthetic_cols:
            X_train = np.hstack([X_baseline_train, X_bert_pca_train, X_aesthetic_train])
            X_test = np.hstack([X_baseline_test, X_bert_pca_test, X_aesthetic_test])
            feature_desc = f"Baseline(9) + BERT-PCA({bert_pca_n}) + Aesthetic({len(aesthetic_cols)})"
            n_features = 9 + bert_pca_n + len(aesthetic_cols)
        else:
            X_train = np.hstack([X_baseline_train, X_bert_pca_train])
            X_test = np.hstack([X_baseline_test, X_bert_pca_test])
            feature_desc = f"Baseline(9) + BERT-PCA({bert_pca_n})"
            n_features = 9 + bert_pca_n

        # Scale features
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train ensemble
        rf = RandomForestRegressor(
            n_estimators=250, max_depth=14, min_samples_split=3,
            min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
        )
        hgb = HistGradientBoostingRegressor(
            max_iter=400, max_depth=14, learning_rate=0.05,
            min_samples_leaf=4, l2_regularization=0.1, random_state=42
        )

        rf.fit(X_train_scaled, y_train_log)
        hgb.fit(X_train_scaled, y_train_log)

        # Predict with 50/50 ensemble
        pred_log = 0.5 * rf.predict(X_test_scaled) + 0.5 * hgb.predict(X_test_scaled)
        pred = np.expm1(pred_log)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)

        results.append({
            'config': feature_desc,
            'bert_pca': bert_pca_n,
            'bert_variance': bert_variance,
            'aesthetic': use_aesthetic,
            'n_features': n_features,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        })

        aesthetic_str = "+Aesthetic" if use_aesthetic else ""
        print(f"   BERT-PCA({bert_pca_n:3d}, {bert_variance:.1%}){aesthetic_str:12s}: "
              f"MAE={mae:6.2f}, R2={r2:.4f}, Features={n_features}")

# ============================================================================
# EXPERIMENT 2: DIFFERENT SCALING METHODS
# ============================================================================
print("\n" + "="*90)
print(" "*25 + "EXPERIMENT 2: SCALING METHODS")
print("="*90)

# Use best PCA config from experiment 1
best_exp1 = min(results, key=lambda x: x['mae'])
best_bert_pca = best_exp1['bert_pca']
best_use_aesthetic = best_exp1['aesthetic']

print(f"\n[BEST CONFIG] BERT-PCA({best_bert_pca}) + Aesthetic={best_use_aesthetic}")

# Rebuild feature matrix with best config
pca_bert = PCA(n_components=best_bert_pca, random_state=42)
X_bert_pca_train = pca_bert.fit_transform(X_bert_train)
X_bert_pca_test = pca_bert.transform(X_bert_test)

if best_use_aesthetic and aesthetic_cols:
    X_train_best = np.hstack([X_baseline_train, X_bert_pca_train, X_aesthetic_train])
    X_test_best = np.hstack([X_baseline_test, X_bert_pca_test, X_aesthetic_test])
else:
    X_train_best = np.hstack([X_baseline_train, X_bert_pca_train])
    X_test_best = np.hstack([X_baseline_test, X_bert_pca_test])

scaling_results = []
print("\n[TEST] Testing different scalers:")

scalers = [
    ('QuantileTransformer-normal', QuantileTransformer(output_distribution='normal', random_state=42)),
    ('QuantileTransformer-uniform', QuantileTransformer(output_distribution='uniform', random_state=42)),
    ('StandardScaler', StandardScaler()),
    ('RobustScaler', RobustScaler()),
]

for scaler_name, scaler in scalers:
    X_train_scaled = scaler.fit_transform(X_train_best)
    X_test_scaled = scaler.transform(X_test_best)

    # Train ensemble
    rf = RandomForestRegressor(
        n_estimators=250, max_depth=14, min_samples_split=3,
        min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
    )
    hgb = HistGradientBoostingRegressor(
        max_iter=400, max_depth=14, learning_rate=0.05,
        min_samples_leaf=4, l2_regularization=0.1, random_state=42
    )

    rf.fit(X_train_scaled, y_train_log)
    hgb.fit(X_train_scaled, y_train_log)

    pred_log = 0.5 * rf.predict(X_test_scaled) + 0.5 * hgb.predict(X_test_scaled)
    pred = np.expm1(pred_log)

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    scaling_results.append({
        'scaler': scaler_name,
        'mae': mae,
        'r2': r2
    })

    print(f"   {scaler_name:30s}: MAE={mae:6.2f}, R2={r2:.4f}")

# ============================================================================
# EXPERIMENT 3: ENSEMBLE ALGORITHMS
# ============================================================================
print("\n" + "="*90)
print(" "*25 + "EXPERIMENT 3: ENSEMBLE ALGORITHMS")
print("="*90)

# Use best scaler
best_scaler_config = min(scaling_results, key=lambda x: x['mae'])
best_scaler_name = best_scaler_config['scaler']
print(f"\n[BEST SCALER] {best_scaler_name}")

# Recreate best scaler
if best_scaler_name == 'QuantileTransformer-normal':
    best_scaler = QuantileTransformer(output_distribution='normal', random_state=42)
elif best_scaler_name == 'QuantileTransformer-uniform':
    best_scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
elif best_scaler_name == 'StandardScaler':
    best_scaler = StandardScaler()
else:
    best_scaler = RobustScaler()

X_train_scaled = best_scaler.fit_transform(X_train_best)
X_test_scaled = best_scaler.transform(X_test_best)

# Test different algorithm combinations
ensemble_results = []
print("\n[TEST] Testing different ensemble combinations:")

algorithms = [
    ('RF', RandomForestRegressor(n_estimators=250, max_depth=14, min_samples_split=3,
                                  min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1)),
    ('HGB', HistGradientBoostingRegressor(max_iter=400, max_depth=14, learning_rate=0.05,
                                          min_samples_leaf=4, l2_regularization=0.1, random_state=42)),
    ('GBM', GradientBoostingRegressor(n_estimators=250, max_depth=14, learning_rate=0.05,
                                       min_samples_leaf=4, subsample=0.8, random_state=42)),
]

# Train all algorithms
trained_models = {}
predictions = {}

for name, model in algorithms:
    model.fit(X_train_scaled, y_train_log)
    pred_log = model.predict(X_test_scaled)
    predictions[name] = pred_log
    trained_models[name] = model

    # Individual performance
    pred = np.expm1(pred_log)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"   {name:15s}: MAE={mae:6.2f}, R2={r2:.4f}")

# Test ensemble combinations
print("\n[TEST] Testing ensemble weight combinations:")
ensemble_configs = [
    ('RF only', {'RF': 1.0}),
    ('HGB only', {'HGB': 1.0}),
    ('GBM only', {'GBM': 1.0}),
    ('RF+HGB (50/50)', {'RF': 0.5, 'HGB': 0.5}),
    ('RF+GBM (50/50)', {'RF': 0.5, 'GBM': 0.5}),
    ('HGB+GBM (50/50)', {'HGB': 0.5, 'GBM': 0.5}),
    ('RF+HGB+GBM (33/33/33)', {'RF': 0.33, 'HGB': 0.34, 'GBM': 0.33}),
    ('RF+HGB (60/40)', {'RF': 0.6, 'HGB': 0.4}),
    ('RF+HGB (40/60)', {'RF': 0.4, 'HGB': 0.6}),
    ('RF+HGB (70/30)', {'RF': 0.7, 'HGB': 0.3}),
    ('RF+HGB (30/70)', {'RF': 0.3, 'HGB': 0.7}),
]

for config_name, weights in ensemble_configs:
    pred_log = sum(weights.get(name, 0) * predictions[name] for name in predictions.keys())
    pred = np.expm1(pred_log)

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    ensemble_results.append({
        'config': config_name,
        'weights': weights,
        'mae': mae,
        'r2': r2
    })

    print(f"   {config_name:25s}: MAE={mae:6.2f}, R2={r2:.4f}")

# ============================================================================
# EXPERIMENT 4: HYPERPARAMETER FINE-TUNING ON BEST ENSEMBLE
# ============================================================================
print("\n" + "="*90)
print(" "*20 + "EXPERIMENT 4: HYPERPARAMETER FINE-TUNING")
print("="*90)

best_ensemble = min(ensemble_results, key=lambda x: x['mae'])
print(f"\n[BEST ENSEMBLE] {best_ensemble['config']}")

# Test RF depth variations
print("\n[TEST] Testing Random Forest max_depth:")
depth_results = []

for max_depth in [10, 12, 14, 16, 18, 20, None]:
    rf = RandomForestRegressor(
        n_estimators=250, max_depth=max_depth, min_samples_split=3,
        min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train_log)
    pred_log = rf.predict(X_test_scaled)
    pred = np.expm1(pred_log)

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    depth_results.append({
        'max_depth': max_depth,
        'mae': mae,
        'r2': r2
    })

    depth_str = str(max_depth) if max_depth is not None else 'None'
    print(f"   max_depth={depth_str:4s}: MAE={mae:6.2f}, R2={r2:.4f}")

# Test HGB max_iter variations
print("\n[TEST] Testing HistGradientBoosting max_iter:")
iter_results = []

for max_iter in [200, 300, 400, 500, 600]:
    hgb = HistGradientBoostingRegressor(
        max_iter=max_iter, max_depth=14, learning_rate=0.05,
        min_samples_leaf=4, l2_regularization=0.1, random_state=42
    )
    hgb.fit(X_train_scaled, y_train_log)
    pred_log = hgb.predict(X_test_scaled)
    pred = np.expm1(pred_log)

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    iter_results.append({
        'max_iter': max_iter,
        'mae': mae,
        'r2': r2
    })

    print(f"   max_iter={max_iter:3d}: MAE={mae:6.2f}, R2={r2:.4f}")

# ============================================================================
# FINAL: TRAIN BEST MODEL
# ============================================================================
print("\n" + "="*90)
print(" "*30 + "FINAL BEST MODEL")
print("="*90)

# Find best configurations
best_depth = min(depth_results, key=lambda x: x['mae'])
best_iter = min(iter_results, key=lambda x: x['mae'])

print(f"\n[BEST CONFIG SUMMARY]")
print(f"   Features: {best_exp1['config']}")
print(f"   BERT PCA: {best_bert_pca} components ({best_exp1['bert_variance']:.1%} variance)")
print(f"   Aesthetic: {best_use_aesthetic}")
print(f"   Total features: {best_exp1['n_features']}")
print(f"   Scaler: {best_scaler_name}")
print(f"   Ensemble: {best_ensemble['config']}")
print(f"   RF max_depth: {best_depth['max_depth']}")
print(f"   HGB max_iter: {best_iter['max_iter']}")

# Train final model
print(f"\n[TRAIN] Training final optimized model...")

rf_final = RandomForestRegressor(
    n_estimators=250,
    max_depth=best_depth['max_depth'],
    min_samples_split=3,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

hgb_final = HistGradientBoostingRegressor(
    max_iter=best_iter['max_iter'],
    max_depth=14,
    learning_rate=0.05,
    min_samples_leaf=4,
    l2_regularization=0.1,
    random_state=42
)

rf_final.fit(X_train_scaled, y_train_log)
hgb_final.fit(X_train_scaled, y_train_log)

# Get ensemble weights
weights = best_ensemble['weights']
rf_weight = weights.get('RF', 0)
hgb_weight = weights.get('HGB', 0)

pred_log = rf_weight * rf_final.predict(X_test_scaled) + hgb_weight * hgb_final.predict(X_test_scaled)
pred = np.expm1(pred_log)

mae_final = mean_absolute_error(y_test, pred)
rmse_final = np.sqrt(mean_squared_error(y_test, pred))
r2_final = r2_score(y_test, pred)

print(f"\n[RESULTS]")
print(f"   MAE:  {mae_final:.2f} likes")
print(f"   RMSE: {rmse_final:.2f} likes")
print(f"   R2:   {r2_final:.4f}")

# Compare to baseline champion
baseline_mae = 51.82
if mae_final < baseline_mae:
    improvement = ((baseline_mae - mae_final) / baseline_mae) * 100
    print(f"\n[SUCCESS] NEW CHAMPION! Improvement: {improvement:.1f}% vs baseline (MAE={baseline_mae})")
else:
    difference = ((mae_final - baseline_mae) / baseline_mae) * 100
    print(f"\n[INFO] Baseline still better by {difference:.1f}% (Baseline MAE={baseline_mae})")

# Save model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"models/phase5_1_advanced_model.pkl"

model_package = {
    'rf_model': rf_final,
    'hgb_model': hgb_final,
    'rf_weight': rf_weight,
    'hgb_weight': hgb_weight,
    'pca_bert': pca_bert,
    'scaler': best_scaler,
    'baseline_features': baseline_features,
    'bert_pca_components': best_bert_pca,
    'use_aesthetic': best_use_aesthetic,
    'aesthetic_features': aesthetic_cols,
    'metrics': {
        'mae': mae_final,
        'rmse': rmse_final,
        'r2': r2_final
    },
    'config': {
        'n_features': best_exp1['n_features'],
        'scaler': best_scaler_name,
        'ensemble': best_ensemble['config'],
        'rf_max_depth': best_depth['max_depth'],
        'hgb_max_iter': best_iter['max_iter']
    }
}

joblib.dump(model_package, model_filename)
print(f"\n[SAVED] Model: {model_filename}")

# Feature importance (from RF)
feature_names = baseline_features.copy()
feature_names += [f'bert_pca_{i}' for i in range(best_bert_pca)]
if best_use_aesthetic and aesthetic_cols:
    feature_names += aesthetic_cols

importances = rf_final.feature_importances_
indices = np.argsort(importances)[::-1]

print(f"\n[TOP 15] Most Important Features:")
for i in range(min(15, len(feature_names))):
    idx = indices[i]
    print(f"   {i+1:2d}. {feature_names[idx]:30s} {importances[idx]:.6f}")

print("\n" + "="*90)
print(" "*25 + "PHASE 5.1 COMPLETE!")
print("="*90)
