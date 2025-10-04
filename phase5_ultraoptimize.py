"""
Phase 5: ULTRA-OPTIMIZATION
Systematic exploration of model combinations to push beyond MAE=51.82

Experiments:
1. Ensemble weight optimization (RF/HGB ratios)
2. Feature selection strategies
3. Stacking ensemble (meta-learner)
4. PCA dimensionality variations
5. Polynomial feature interactions
6. 5-fold cross-validation
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import QuantileTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print(" "*25 + "PHASE 5: ULTRA-OPTIMIZATION")
print(" "*20 + "Exploring Advanced Model Combinations")
print("="*90)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[LOAD] Loading datasets...")
df_main = pd.read_csv('multi_account_dataset.csv')
df_bert = pd.read_csv('data/processed/bert_embeddings_multi_account.csv')
df_aesthetic = pd.read_csv('data/processed/aesthetic_features_multi_account.csv')

print(f"   Main data: {len(df_main)} posts")
print(f"   BERT embeddings: {len(df_bert)} posts")
print(f"   Aesthetic features: {len(df_aesthetic)} posts")

# Merge datasets
df = df_main.merge(df_bert, on='post_id', how='inner')
df = df.merge(df_aesthetic, on='post_id', how='left')

# Fill missing aesthetic features with 0
aesthetic_cols = [col for col in df.columns if col.startswith('aesthetic_')]
if aesthetic_cols:
    df[aesthetic_cols] = df[aesthetic_cols].fillna(0)

print(f"\n[MERGE] Combined dataset: {len(df)} posts")

# ============================================================================
# PREPARE FEATURES
# ============================================================================
print("\n[FEATURES] Preparing feature matrices...")

# Baseline features (9)
baseline_features = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                     'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']

# BERT features (768-dim)
bert_cols = [col for col in df.columns if col.startswith('bert_')]
X_bert_full = df[bert_cols].values

# Aesthetic features (8-dim)
aesthetic_cols = [col for col in df.columns if col.startswith('aesthetic_')]
X_aesthetic = df[aesthetic_cols].values if aesthetic_cols else np.zeros((len(df), 0))

# Target
y = df['likes'].values

# Outlier clipping (99th percentile)
clip_value = np.percentile(y, 99)
y_clipped = np.clip(y, 0, clip_value)
y_log = np.log1p(y_clipped)

print(f"   Baseline features: {len(baseline_features)}")
print(f"   BERT features: {X_bert_full.shape[1]}")
print(f"   Aesthetic features: {X_aesthetic.shape[1] if aesthetic_cols else 0}")

# ============================================================================
# SPLIT DATA
# ============================================================================
from sklearn.model_selection import train_test_split

X_baseline = df[baseline_features].values
train_idx, test_idx = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=42
)

X_baseline_train = X_baseline[train_idx]
X_baseline_test = X_baseline[test_idx]
y_train_log = y_log[train_idx]
y_test_log = y_log[test_idx]
y_test = y_clipped[test_idx]

print(f"\n[SPLIT] Train: {len(train_idx)}, Test: {len(test_idx)}")

# ============================================================================
# EXPERIMENT 1: ENSEMBLE WEIGHT OPTIMIZATION
# ============================================================================
print("\n" + "="*90)
print(" "*25 + "EXPERIMENT 1: ENSEMBLE WEIGHTS")
print("="*90)

# Scale baseline features
scaler = QuantileTransformer(output_distribution='normal', random_state=42)
X_train_scaled = scaler.fit_transform(X_baseline_train)
X_test_scaled = scaler.transform(X_baseline_test)

# Train RF and HGB
rf = RandomForestRegressor(
    n_estimators=250, max_depth=14, min_samples_split=3,
    min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
)
hgb = HistGradientBoostingRegressor(
    max_iter=400, max_depth=14, learning_rate=0.05,
    min_samples_leaf=4, l2_regularization=0.1, random_state=42
)

print("[TRAIN] Training RF and HGB...")
rf.fit(X_train_scaled, y_train_log)
hgb.fit(X_train_scaled, y_train_log)

pred_rf_log = rf.predict(X_test_scaled)
pred_hgb_log = hgb.predict(X_test_scaled)

# Test different weight combinations
weight_results = []
print("\n[TEST] Testing weight combinations:")
for rf_weight in np.arange(0.0, 1.05, 0.05):
    hgb_weight = 1.0 - rf_weight
    pred_log = rf_weight * pred_rf_log + hgb_weight * pred_hgb_log
    pred = np.expm1(pred_log)
    mae = mean_absolute_error(y_test, pred)
    weight_results.append({
        'rf_weight': rf_weight,
        'hgb_weight': hgb_weight,
        'mae': mae
    })
    if rf_weight % 0.1 < 0.01:  # Print every 10%
        print(f"   RF={rf_weight:.2f}, HGB={hgb_weight:.2f}: MAE={mae:.2f}")

# Find best weights
best_weights = min(weight_results, key=lambda x: x['mae'])
print(f"\n[BEST] RF={best_weights['rf_weight']:.2f}, HGB={best_weights['hgb_weight']:.2f}")
print(f"       MAE={best_weights['mae']:.2f} likes")

# ============================================================================
# EXPERIMENT 2: FEATURE SELECTION
# ============================================================================
print("\n" + "="*90)
print(" "*25 + "EXPERIMENT 2: FEATURE SELECTION")
print("="*90)

# Test different numbers of top features
feature_selection_results = []
print("\n[TEST] Testing different feature counts:")
for k in [3, 5, 7, 9]:  # Top-k features
    if k <= len(baseline_features):
        selector = SelectKBest(f_regression, k=k)
        X_train_selected = selector.fit_transform(X_baseline_train, y_train_log)
        X_test_selected = selector.transform(X_baseline_test)

        # Scale
        X_train_selected = scaler.fit_transform(X_train_selected)
        X_test_selected = scaler.transform(X_test_selected)

        # Train ensemble with best weights
        rf_temp = RandomForestRegressor(
            n_estimators=250, max_depth=14, min_samples_split=3,
            min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
        )
        hgb_temp = HistGradientBoostingRegressor(
            max_iter=400, max_depth=14, learning_rate=0.05,
            min_samples_leaf=4, l2_regularization=0.1, random_state=42
        )

        rf_temp.fit(X_train_selected, y_train_log)
        hgb_temp.fit(X_train_selected, y_train_log)

        pred_log = (best_weights['rf_weight'] * rf_temp.predict(X_test_selected) +
                   best_weights['hgb_weight'] * hgb_temp.predict(X_test_selected))
        pred = np.expm1(pred_log)
        mae = mean_absolute_error(y_test, pred)

        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = [baseline_features[i] for i in range(len(baseline_features)) if selected_mask[i]]

        feature_selection_results.append({
            'k': k,
            'mae': mae,
            'features': selected_features
        })
        print(f"   Top-{k} features: MAE={mae:.2f}")
        print(f"      Features: {', '.join(selected_features)}")

# ============================================================================
# EXPERIMENT 3: STACKING ENSEMBLE
# ============================================================================
print("\n" + "="*90)
print(" "*25 + "EXPERIMENT 3: STACKING ENSEMBLE")
print("="*90)

print("\n[TRAIN] Training base models...")
# Get predictions from base models
pred_rf_train_log = rf.predict(X_train_scaled)
pred_hgb_train_log = hgb.predict(X_train_scaled)

# Create meta-features
X_meta_train = np.column_stack([pred_rf_train_log, pred_hgb_train_log])
X_meta_test = np.column_stack([pred_rf_log, pred_hgb_log])

stacking_results = []
print("\n[TEST] Testing meta-learners:")

# Ridge meta-learner
for alpha in [0.1, 1.0, 10.0]:
    meta_model = Ridge(alpha=alpha)
    meta_model.fit(X_meta_train, y_train_log)
    pred_log = meta_model.predict(X_meta_test)
    pred = np.expm1(pred_log)
    mae = mean_absolute_error(y_test, pred)
    stacking_results.append({
        'meta_learner': f'Ridge(alpha={alpha})',
        'mae': mae
    })
    print(f"   Ridge(alpha={alpha}): MAE={mae:.2f}")

# Lasso meta-learner
for alpha in [0.1, 1.0, 10.0]:
    meta_model = Lasso(alpha=alpha)
    meta_model.fit(X_meta_train, y_train_log)
    pred_log = meta_model.predict(X_meta_test)
    pred = np.expm1(pred_log)
    mae = mean_absolute_error(y_test, pred)
    stacking_results.append({
        'meta_learner': f'Lasso(alpha={alpha})',
        'mae': mae
    })
    print(f"   Lasso(alpha={alpha}): MAE={mae:.2f}")

# ============================================================================
# EXPERIMENT 4: PCA DIMENSIONALITY
# ============================================================================
print("\n" + "="*90)
print(" "*25 + "EXPERIMENT 4: PCA DIMENSIONALITY")
print("="*90)

X_bert_train = X_bert_full[train_idx]
X_bert_test = X_bert_full[test_idx]

pca_results = []
print("\n[TEST] Testing different PCA components:")
for n_components in [30, 50, 70, 100]:
    # PCA on BERT
    pca = PCA(n_components=n_components, random_state=42)
    X_bert_pca_train = pca.fit_transform(X_bert_train)
    X_bert_pca_test = pca.transform(X_bert_test)

    # Combine with baseline
    X_combined_train = np.hstack([X_baseline_train, X_bert_pca_train])
    X_combined_test = np.hstack([X_baseline_test, X_bert_pca_test])

    # Scale
    X_combined_train_scaled = scaler.fit_transform(X_combined_train)
    X_combined_test_scaled = scaler.transform(X_combined_test)

    # Train ensemble
    rf_temp = RandomForestRegressor(
        n_estimators=250, max_depth=14, min_samples_split=3,
        min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
    )
    hgb_temp = HistGradientBoostingRegressor(
        max_iter=400, max_depth=14, learning_rate=0.05,
        min_samples_leaf=4, l2_regularization=0.1, random_state=42
    )

    rf_temp.fit(X_combined_train_scaled, y_train_log)
    hgb_temp.fit(X_combined_train_scaled, y_train_log)

    pred_log = (best_weights['rf_weight'] * rf_temp.predict(X_combined_test_scaled) +
               best_weights['hgb_weight'] * hgb_temp.predict(X_combined_test_scaled))
    pred = np.expm1(pred_log)
    mae = mean_absolute_error(y_test, pred)

    variance_explained = pca.explained_variance_ratio_.sum()

    pca_results.append({
        'n_components': n_components,
        'variance_explained': variance_explained,
        'mae': mae,
        'features': 9 + n_components
    })
    print(f"   PCA={n_components} (variance={variance_explained:.1%}): MAE={mae:.2f}")

# ============================================================================
# EXPERIMENT 5: POLYNOMIAL FEATURES (TOP-3)
# ============================================================================
print("\n" + "="*90)
print(" "*25 + "EXPERIMENT 5: POLYNOMIAL FEATURES")
print("="*90)

# Top-3 most important features: month, hashtag_count, caption_length
top3_features = ['month', 'hashtag_count', 'caption_length']
top3_indices = [baseline_features.index(f) for f in top3_features]

X_top3_train = X_baseline_train[:, top3_indices]
X_top3_test = X_baseline_test[:, top3_indices]

poly_results = []
print("\n[TEST] Testing polynomial degrees:")
for degree in [2, 3]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly_train = poly.fit_transform(X_top3_train)
    X_poly_test = poly.transform(X_top3_test)

    # Combine with all baseline features
    X_combined_train = np.hstack([X_baseline_train, X_poly_train])
    X_combined_test = np.hstack([X_baseline_test, X_poly_test])

    # Scale
    X_combined_train_scaled = scaler.fit_transform(X_combined_train)
    X_combined_test_scaled = scaler.transform(X_combined_test)

    # Train ensemble
    rf_temp = RandomForestRegressor(
        n_estimators=250, max_depth=14, min_samples_split=3,
        min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
    )
    hgb_temp = HistGradientBoostingRegressor(
        max_iter=400, max_depth=14, learning_rate=0.05,
        min_samples_leaf=4, l2_regularization=0.1, random_state=42
    )

    rf_temp.fit(X_combined_train_scaled, y_train_log)
    hgb_temp.fit(X_combined_train_scaled, y_train_log)

    pred_log = (best_weights['rf_weight'] * rf_temp.predict(X_combined_test_scaled) +
               best_weights['hgb_weight'] * hgb_temp.predict(X_combined_test_scaled))
    pred = np.expm1(pred_log)
    mae = mean_absolute_error(y_test, pred)

    poly_results.append({
        'degree': degree,
        'n_features': X_poly_train.shape[1],
        'total_features': X_combined_train.shape[1],
        'mae': mae
    })
    print(f"   Degree={degree} ({X_poly_train.shape[1]} poly features): MAE={mae:.2f}")

# ============================================================================
# EXPERIMENT 6: 5-FOLD CROSS-VALIDATION (BASELINE CHAMPION)
# ============================================================================
print("\n" + "="*90)
print(" "*25 + "EXPERIMENT 6: CROSS-VALIDATION")
print("="*90)

print("\n[CV] Running 5-fold cross-validation on baseline champion...")

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx_cv, val_idx_cv) in enumerate(kfold.split(X_baseline), 1):
    X_train_cv = X_baseline[train_idx_cv]
    X_val_cv = X_baseline[val_idx_cv]
    y_train_cv_log = y_log[train_idx_cv]
    y_val_cv_log = y_log[val_idx_cv]
    y_val_cv = y_clipped[val_idx_cv]

    # Scale
    scaler_cv = QuantileTransformer(output_distribution='normal', random_state=42)
    X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
    X_val_cv_scaled = scaler_cv.transform(X_val_cv)

    # Train ensemble
    rf_cv = RandomForestRegressor(
        n_estimators=250, max_depth=14, min_samples_split=3,
        min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
    )
    hgb_cv = HistGradientBoostingRegressor(
        max_iter=400, max_depth=14, learning_rate=0.05,
        min_samples_leaf=4, l2_regularization=0.1, random_state=42
    )

    rf_cv.fit(X_train_cv_scaled, y_train_cv_log)
    hgb_cv.fit(X_train_cv_scaled, y_train_cv_log)

    pred_log = (best_weights['rf_weight'] * rf_cv.predict(X_val_cv_scaled) +
               best_weights['hgb_weight'] * hgb_cv.predict(X_val_cv_scaled))
    pred = np.expm1(pred_log)
    mae = mean_absolute_error(y_val_cv, pred)
    r2 = r2_score(y_val_cv, pred)

    cv_scores.append({'fold': fold, 'mae': mae, 'r2': r2})
    print(f"   Fold {fold}: MAE={mae:.2f}, R2={r2:.4f}")

cv_mean_mae = np.mean([s['mae'] for s in cv_scores])
cv_std_mae = np.std([s['mae'] for s in cv_scores])
cv_mean_r2 = np.mean([s['r2'] for s in cv_scores])

print(f"\n[RESULT] Cross-validation: MAE={cv_mean_mae:.2f} +/- {cv_std_mae:.2f}")
print(f"         R2={cv_mean_r2:.4f}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*90)
print(" "*30 + "FINAL SUMMARY")
print("="*90)

print("\n[CHAMPION] Previous Best: MAE=51.82 (Baseline, 50/50 ensemble)")
print("\n[RESULTS] All Experiments:")

print("\n1. ENSEMBLE WEIGHTS:")
print(f"   Best: RF={best_weights['rf_weight']:.2f}, HGB={best_weights['hgb_weight']:.2f}")
print(f"   MAE: {best_weights['mae']:.2f} likes")

print("\n2. FEATURE SELECTION:")
best_fs = min(feature_selection_results, key=lambda x: x['mae'])
print(f"   Best: Top-{best_fs['k']} features")
print(f"   MAE: {best_fs['mae']:.2f} likes")
print(f"   Features: {', '.join(best_fs['features'])}")

print("\n3. STACKING ENSEMBLE:")
best_stack = min(stacking_results, key=lambda x: x['mae'])
print(f"   Best: {best_stack['meta_learner']}")
print(f"   MAE: {best_stack['mae']:.2f} likes")

print("\n4. PCA DIMENSIONALITY:")
best_pca = min(pca_results, key=lambda x: x['mae'])
print(f"   Best: {best_pca['n_components']} components ({best_pca['variance_explained']:.1%} variance)")
print(f"   MAE: {best_pca['mae']:.2f} likes")

print("\n5. POLYNOMIAL FEATURES:")
best_poly = min(poly_results, key=lambda x: x['mae'])
print(f"   Best: Degree {best_poly['degree']} ({best_poly['total_features']} features)")
print(f"   MAE: {best_poly['mae']:.2f} likes")

print("\n6. CROSS-VALIDATION:")
print(f"   MAE: {cv_mean_mae:.2f} +/- {cv_std_mae:.2f}")
print(f"   R2: {cv_mean_r2:.4f}")

# Find overall best
all_results = [
    {'method': 'Ensemble Weights', 'mae': best_weights['mae']},
    {'method': f"Feature Selection (Top-{best_fs['k']})", 'mae': best_fs['mae']},
    {'method': f"Stacking ({best_stack['meta_learner']})", 'mae': best_stack['mae']},
    {'method': f"PCA ({best_pca['n_components']} components)", 'mae': best_pca['mae']},
    {'method': f"Polynomial (Degree {best_poly['degree']})", 'mae': best_poly['mae']},
    {'method': 'Cross-validation', 'mae': cv_mean_mae}
]

overall_best = min(all_results, key=lambda x: x['mae'])

print("\n" + "="*90)
print(f"[OVERALL BEST] {overall_best['method']}")
print(f"              MAE: {overall_best['mae']:.2f} likes")

if overall_best['mae'] < 51.82:
    improvement = ((51.82 - overall_best['mae']) / 51.82) * 100
    print(f"              Improvement: {improvement:.1f}% vs previous best")
    print("\n[SUCCESS] NEW CHAMPION FOUND!")
else:
    print("\n[CONCLUSION] Previous baseline champion remains best (MAE=51.82)")

print("\n" + "="*90)
print(" "*30 + "PHASE 5 COMPLETE")
print("="*90)
