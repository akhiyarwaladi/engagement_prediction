#!/usr/bin/env python3
"""
PRODUCTION MODEL: Simple Temporal Features
Based on ablation study findings - simple features outperform complex ones!
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime


def load_and_prepare_data():
    """Load data and create simple temporal features"""
    print("\n[DATA] Loading dataset...")

    df = pd.read_csv('fst_unja_from_gallery_dl.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"  Loaded {len(df)} posts")

    # Baseline features
    df['caption'] = df['caption'].fillna('')
    df['caption_length'] = df['caption'].str.len()
    df['word_count'] = df['caption'].str.split().str.len()
    df['hashtag_count'] = df['hashtags_count']
    df['mention_count'] = df['mentions_count']
    df['is_video'] = df['is_video'].astype(int)
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)

    # Cyclic temporal features
    hour = df['date'].dt.hour
    day = df['date'].dt.dayofweek
    month = df['date'].dt.month

    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * day / 7)
    df['day_cos'] = np.cos(2 * np.pi * day / 7)
    df['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)

    # Lag features
    for lag in [1, 2, 3, 5]:
        df[f'likes_lag_{lag}'] = df['likes'].shift(lag).fillna(df['likes'].median())

    df['likes_rolling_mean_5'] = df['likes'].rolling(5, min_periods=1).mean()
    df['likes_rolling_std_5'] = df['likes'].rolling(5, min_periods=1).std().fillna(0)

    # Select features
    features = [
        # Baseline (6)
        'caption_length', 'word_count', 'hashtag_count', 'mention_count',
        'is_video', 'is_weekend',
        # Cyclic (6)
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        # Lag (6)
        'likes_lag_1', 'likes_lag_2', 'likes_lag_3', 'likes_lag_5',
        'likes_rolling_mean_5', 'likes_rolling_std_5'
    ]

    X = df[features]
    y = df['likes'].values

    print(f"\n[FEATURES] Total features: {len(features)}")
    print(f"  Baseline: 6")
    print(f"  Cyclic temporal: 6")
    print(f"  Lag features: 6")

    return X, y, df


def cross_validate_model(X, y, n_splits=5):
    """Perform time-series cross-validation"""
    print(f"\n[CV] Running {n_splits}-fold time-series cross-validation...")

    # Import StackingRegressor
    from sklearn.ensemble import StackingRegressor, HistGradientBoostingRegressor

    # Models to test (using ablation study best config)
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=300,
            max_depth=26,
            min_samples_split=2,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Stacking (GB meta)': StackingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(
                    n_estimators=300,
                    max_depth=26,
                    random_state=42,
                    n_jobs=-1
                )),
                ('hgb', HistGradientBoostingRegressor(
                    max_iter=254,
                    max_depth=15,
                    learning_rate=0.104,
                    random_state=42
                ))
            ],
            final_estimator=GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                random_state=42
            ),
            cv=5
        )
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = {}

    for name, model in models.items():
        print(f"\n  Testing {name}...")

        # CV scores
        cv_scores = cross_val_score(
            model, X, y,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )

        mae_scores = -cv_scores
        mean_mae = mae_scores.mean()
        std_mae = mae_scores.std()

        print(f"    MAE: {mean_mae:.2f} ± {std_mae:.2f}")
        print(f"    Folds: {[f'{s:.2f}' for s in mae_scores]}")

        results[name] = {
            'mean_mae': mean_mae,
            'std_mae': std_mae,
            'cv_scores': mae_scores
        }

    # Find best
    best_model_name = min(results.items(), key=lambda x: x[1]['mean_mae'])[0]
    print(f"\n  [BEST] {best_model_name}")
    print(f"    CV MAE: {results[best_model_name]['mean_mae']:.2f} ± {results[best_model_name]['std_mae']:.2f}")

    return models[best_model_name], results


def train_final_model(X, y, model, test_size=0.2):
    """Train final model on full training set"""
    print("\n[TRAIN] Training final production model...")

    # Time-based split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"\n[RESULTS]")
    print(f"  Train MAE: {train_mae:.2f}  R²: {train_r2:.3f}")
    print(f"  Test  MAE: {test_mae:.2f}  R²: {test_r2:.3f}")
    print(f"  Overfit Gap: {abs(train_r2 - test_r2):.3f}")

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print(f"\n[FEATURE IMPORTANCE] Top 10:")
        for _, row in feature_imp.head(10).iterrows():
            print(f"  {row['feature']:25s} {row['importance']:.4f}")

    return model, {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2
    }


def save_production_model(model, metrics, X):
    """Save production model"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'models/production_simple_temporal_{timestamp}.pkl'

    model_data = {
        'model': model,
        'feature_names': X.columns.tolist(),
        'metrics': metrics,
        'model_type': type(model).__name__,
        'training_date': timestamp,
        'n_features': len(X.columns),
        'description': 'Production model: Simple temporal features (baseline + cyclic + lag)'
    }

    joblib.dump(model_data, model_path)
    print(f"\n[SAVED] Production model saved to: {model_path}")

    return model_path


def main():
    """Main execution"""
    print("=" * 80)
    print("PRODUCTION MODEL TRAINING")
    print("Simple Temporal Features (Based on Ablation Study)")
    print("=" * 80)

    # Load data
    X, y, df = load_and_prepare_data()

    # Cross-validation
    best_model, cv_results = cross_validate_model(X, y, n_splits=5)

    # Train final model
    final_model, metrics = train_final_model(X, y, best_model)

    # Save
    model_path = save_production_model(final_model, metrics, X)

    # Summary
    print("\n" + "=" * 80)
    print("PRODUCTION MODEL READY")
    print("=" * 80)

    # Find model name in cv_results
    model_name = None
    model_type = type(final_model).__name__
    for name in cv_results.keys():
        if name in model_type or model_type.replace('Regressor', '') in name or 'Stacking' in name:
            model_name = name
            break

    print(f"\nModel: {type(final_model).__name__}")
    print(f"Features: {len(X.columns)}")
    print(f"Test MAE: {metrics['test_mae']:.2f} likes")
    print(f"Test R²: {metrics['test_r2']:.3f}")

    if model_name:
        print(f"CV MAE: {cv_results[model_name]['mean_mae']:.2f} ± {cv_results[model_name]['std_mae']:.2f}")

    print(f"\nSaved to: {model_path}")
    print("\nREADY FOR PRODUCTION DEPLOYMENT!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
