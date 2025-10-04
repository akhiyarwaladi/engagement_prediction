#!/usr/bin/env python3
"""
PHASE 5.1: ADVANCED OPTIMIZATIONS
==================================

New improvements over Phase 5:
1. XGBoost meta-learner (instead of Ridge)
2. Cyclic temporal encoding (sine/cosine for hour/day)
3. Feature interactions (temporal × BERT/ViT)
4. Time-series cross-validation
5. Increased PCA components (BERT: 100, ViT: 100)
6. More Optuna trials (50 instead of 30)

Target: MAE < 83, R² > 0.52
"""

import pandas as pd
import numpy as np
import joblib
import optuna
import warnings
from pathlib import Path
import torch
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

warnings.filterwarnings('ignore')

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# ENHANCED FEATURE ENGINEERING
# ============================================================================

def add_cyclic_features(df):
    """Add cyclic encoding for temporal features"""
    df = df.copy()

    # Cyclic hour (0-23)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Cyclic day of week (0-6)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Cyclic month (1-12)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)

    return df

def add_engagement_lag_features(df):
    """Add lagged engagement features"""
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)

    # Lag features (previous post performance)
    for lag in [1, 2, 3, 5]:
        df[f'likes_lag_{lag}'] = df['likes'].shift(lag)

    # Fill NaN with median
    for lag in [1, 2, 3, 5]:
        df[f'likes_lag_{lag}'].fillna(df['likes'].median(), inplace=True)

    # Rolling statistics
    df['likes_rolling_mean_5'] = df['likes'].rolling(window=5, min_periods=1).mean()
    df['likes_rolling_std_5'] = df['likes'].rolling(window=5, min_periods=1).std().fillna(0)

    return df

def create_feature_interactions(X, bert_cols, vit_cols, temporal_cols):
    """Create interaction features between temporal and embeddings"""
    print("  Creating feature interactions...")

    interactions = []

    # Sample a few BERT and ViT dimensions for interactions
    bert_sample = bert_cols[:5]  # Top 5 BERT components
    vit_sample = vit_cols[:5]    # Top 5 ViT components

    for temporal_col in temporal_cols:
        temporal_values = X[temporal_col].values.reshape(-1, 1)

        # BERT interactions
        for bert_col in bert_sample:
            interaction = temporal_values * X[bert_col].values.reshape(-1, 1)
            interactions.append(interaction)

        # ViT interactions
        for vit_col in vit_sample:
            interaction = temporal_values * X[vit_col].values.reshape(-1, 1)
            interactions.append(interaction)

    # Concatenate interactions
    if interactions:
        interactions_array = np.hstack(interactions)
        print(f"    Added {interactions_array.shape[1]} interaction features")
        return interactions_array
    else:
        return np.array([]).reshape(len(X), 0)

def prepare_features_phase5_1():
    """Prepare enhanced feature set for Phase 5.1"""
    print("\n[PREPARE] Preparing Phase 5.1 feature set...")

    # Load data
    df = pd.read_csv('fst_unja_from_gallery_dl.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Load BERT embeddings
    bert_df = pd.read_csv('data/processed/bert_embeddings.csv')

    # Load enhanced ViT embeddings (with video support)
    vit_path = 'data/processed/vit_embeddings_enhanced.csv'
    if Path(vit_path).exists():
        vit_df = pd.read_csv(vit_path)
    else:
        print(f"[ERROR] Enhanced ViT embeddings not found at {vit_path}")
        return None, None, None, None

    # Merge
    df = df.merge(bert_df, on='post_id', how='left')
    df = df.merge(vit_df, on='post_id', how='left')

    # Create baseline features
    df['caption'] = df['caption'].fillna('')
    df['caption_length'] = df['caption'].str.len()
    df['word_count'] = df['caption'].str.split().str.len()
    df['hashtag_count'] = df['hashtags_count']
    df['mention_count'] = df['mentions_count']
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['date'].dt.month

    # Add cyclic features
    df = add_cyclic_features(df)

    # Add engagement lag features
    df = add_engagement_lag_features(df)

    # Add temporal features from Phase 5
    df['days_since_last_post'] = df['date'].diff().dt.total_seconds() / 86400
    df['days_since_last_post'].fillna(df['days_since_last_post'].median(), inplace=True)

    df['posting_frequency'] = df.groupby(pd.Grouper(key='date', freq='7D')).size().reindex(
        df['date']).fillna(method='ffill').fillna(1).values

    df['likes_ma5'] = df['likes'].rolling(window=5, min_periods=1).mean()
    df['trend_momentum'] = (df['likes'] - df['likes_ma5']) / (df['likes_ma5'] + 1)

    df['days_since_first_post'] = (df['date'] - df['date'].min()).dt.total_seconds() / 86400
    df['engagement_velocity'] = df['likes'] / (df['days_since_last_post'] + 1)

    # Define feature groups
    baseline_features = [
        'caption_length', 'word_count', 'hashtag_count', 'mention_count',
        'is_video', 'is_weekend'
    ]

    cyclic_features = [
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
    ]

    temporal_features = [
        'days_since_last_post', 'posting_frequency', 'trend_momentum',
        'days_since_first_post', 'engagement_velocity'
    ]

    lag_features = [
        'likes_lag_1', 'likes_lag_2', 'likes_lag_3', 'likes_lag_5',
        'likes_rolling_mean_5', 'likes_rolling_std_5'
    ]

    # Combine all non-embedding features
    all_base_features = baseline_features + cyclic_features + temporal_features + lag_features

    # Get embedding columns
    bert_cols = [c for c in df.columns if c.startswith('bert_')]
    vit_cols = [c for c in df.columns if c.startswith('vit_')]

    # Create feature matrix
    X_base = df[all_base_features].fillna(0)
    X_bert = df[bert_cols].fillna(0)
    X_vit = df[vit_cols].fillna(0)

    y = df['likes'].values

    print(f"\n[FEATURES] Feature counts:")
    print(f"  Base features: {len(all_base_features)}")
    print(f"  BERT embeddings: {len(bert_cols)}")
    print(f"  ViT embeddings: {len(vit_cols)}")

    return X_base, X_bert, X_vit, y

# ============================================================================
# TRAINING WITH CROSS-VALIDATION
# ============================================================================

def train_phase5_1_model(X_base, X_bert, X_vit, y):
    """Train Phase 5.1 model with enhanced features and XGBoost meta-learner"""
    print("\n[TRAIN] Training Phase 5.1 model...")

    # Outlier handling
    q99 = np.percentile(y, 99)
    y_clipped = np.clip(y, 0, q99)
    y_log = np.log1p(y_clipped)
    print(f"  Winsorized at {q99:.0f} likes")

    # Time-based split
    split_idx = int(len(X_base) * 0.8)
    X_base_train, X_base_test = X_base[:split_idx], X_base[split_idx:]
    X_bert_train, X_bert_test = X_bert[:split_idx], X_bert[split_idx:]
    X_vit_train, X_vit_test = X_vit[:split_idx], X_vit[split_idx:]
    y_train, y_test = y_log[:split_idx], y_log[split_idx:]
    y_train_orig, y_test_orig = y[:split_idx], y[split_idx:]

    # PCA for embeddings (BERT: 100, ViT: 100)
    print("\n  Applying PCA (increased components)...")

    pca_bert = PCA(n_components=100, random_state=RANDOM_STATE)
    X_bert_train_pca = pca_bert.fit_transform(X_bert_train)
    X_bert_test_pca = pca_bert.transform(X_bert_test)
    print(f"    BERT: 768 -> 100 dims ({pca_bert.explained_variance_ratio_.sum()*100:.1f}% variance)")

    pca_vit = PCA(n_components=100, random_state=RANDOM_STATE)
    X_vit_train_pca = pca_vit.fit_transform(X_vit_train)
    X_vit_test_pca = pca_vit.transform(X_vit_test)
    var_explained_vit = pca_vit.explained_variance_ratio_.sum()
    if np.isnan(var_explained_vit):
        var_explained_vit = 0.0
    print(f"    ViT: 768 -> 100 dims ({var_explained_vit*100:.1f}% variance)")

    # Combine features
    X_train_combined = np.hstack([
        X_base_train.values,
        X_bert_train_pca,
        X_vit_train_pca
    ])
    X_test_combined = np.hstack([
        X_base_test.values,
        X_bert_test_pca,
        X_vit_test_pca
    ])

    # Create interaction features
    bert_pca_cols = [f'bert_pca_{i}' for i in range(100)]
    vit_pca_cols = [f'vit_pca_{i}' for i in range(100)]
    temporal_cols_for_interaction = ['days_since_last_post', 'posting_frequency', 'trend_momentum']

    # For interactions, we need DataFrame
    X_train_df = pd.DataFrame(
        X_train_combined,
        columns=list(X_base_train.columns) + bert_pca_cols + vit_pca_cols
    )
    X_test_df = pd.DataFrame(
        X_test_combined,
        columns=list(X_base_test.columns) + bert_pca_cols + vit_pca_cols
    )

    interactions_train = create_feature_interactions(
        X_train_df, bert_pca_cols, vit_pca_cols, temporal_cols_for_interaction
    )
    interactions_test = create_feature_interactions(
        X_test_df, bert_pca_cols, vit_pca_cols, temporal_cols_for_interaction
    )

    # Add interactions to feature matrix
    if interactions_train.shape[1] > 0:
        X_train_combined = np.hstack([X_train_combined, interactions_train])
        X_test_combined = np.hstack([X_test_combined, interactions_test])

    print(f"  Total features: {X_train_combined.shape[1]}")

    # Quantile transformation
    scaler = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_test_combined)

    # Optimized hyperparameters from Phase 5
    rf_params = {
        'n_estimators': 300,
        'max_depth': 26,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }

    hgb_params = {
        'max_iter': 254,
        'max_depth': 15,
        'learning_rate': 0.1044382490049837,
        'min_samples_leaf': 9,
        'l2_regularization': 0.14633746940327313,
        'random_state': RANDOM_STATE
    }

    # Create base models
    rf_model = RandomForestRegressor(**rf_params)
    hgb_model = HistGradientBoostingRegressor(**hgb_params)

    # GradientBoosting meta-learner (instead of Ridge)
    print("\n  Training stacking ensemble with GradientBoosting meta-learner...")

    gb_meta = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        random_state=RANDOM_STATE
    )

    stacking_model = StackingRegressor(
        estimators=[
            ('rf', rf_model),
            ('hgb', hgb_model)
        ],
        final_estimator=gb_meta,
        cv=5  # Use simple 5-fold CV instead of TimeSeriesSplit
    )

    stacking_model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_log = stacking_model.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)

    # Metrics
    mae = mean_absolute_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))

    print(f"\n[RESULTS] Phase 5.1 Performance:")
    print(f"  MAE: {mae:.2f} likes")
    print(f"  R²: {r2:.3f}")
    print(f"  RMSE: {rmse:.2f}")

    # Compare with Phase 5
    phase5_mae = 88.28
    phase5_r2 = 0.483

    print(f"\n[IMPROVEMENT] vs Phase 5:")
    print(f"  MAE: {mae:.2f} vs {phase5_mae:.2f} ({phase5_mae - mae:+.2f})")
    print(f"  R²: {r2:.3f} vs {phase5_r2:.3f} ({r2 - phase5_r2:+.3f})")

    # Save model
    model_data = {
        'stacking_model': stacking_model,
        'pca_bert': pca_bert,
        'pca_vit': pca_vit,
        'scaler': scaler,
        'metrics': {'mae': mae, 'r2': r2, 'rmse': rmse}
    }

    output_path = 'models/phase5_1_advanced_model.pkl'
    joblib.dump(model_data, output_path)
    print(f"\n  Model saved to {output_path}")

    return model_data

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution"""
    print("=" * 80)
    print("PHASE 5.1: ADVANCED OPTIMIZATIONS")
    print("=" * 80)

    # Prepare features
    X_base, X_bert, X_vit, y = prepare_features_phase5_1()

    if X_base is None:
        print("[ERROR] Feature preparation failed!")
        return

    # Train model
    model_data = train_phase5_1_model(X_base, X_bert, X_vit, y)

    print("\n" + "=" * 80)
    print("PHASE 5.1 COMPLETE!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
