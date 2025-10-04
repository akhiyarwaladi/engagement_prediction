#!/usr/bin/env python3
"""
HYPERPARAMETER SEARCH: Find optimal hyperparameters
Uses Optuna for Bayesian optimization
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from src.training.experiment import ExperimentConfig, get_tracker
from src.training.trainer import run_experiment, ModelTrainer


def load_full_dataset():
    """Load full dataset with all features"""
    print("\n[DATA] Loading dataset...")

    df = pd.read_csv('fst_unja_from_gallery_dl.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Load embeddings
    bert_df = pd.read_csv('data/processed/bert_embeddings.csv')
    vit_df = pd.read_csv('data/processed/vit_embeddings_enhanced.csv')

    df = df.merge(bert_df, on='post_id', how='left')
    df = df.merge(vit_df, on='post_id', how='left')

    # Create features
    df['caption'] = df['caption'].fillna('')
    df['caption_length'] = df['caption'].str.len()
    df['word_count'] = df['caption'].str.split().str.len()
    df['hashtag_count'] = df['hashtags_count']
    df['mention_count'] = df['mentions_count']
    df['is_video'] = df['is_video'].astype(int)
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)

    # Cyclic
    hour = df['date'].dt.hour
    day = df['date'].dt.dayofweek
    month = df['date'].dt.month

    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * day / 7)
    df['day_cos'] = np.cos(2 * np.pi * day / 7)
    df['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)

    # Lag
    for lag in [1, 2, 3, 5]:
        df[f'likes_lag_{lag}'] = df['likes'].shift(lag).fillna(df['likes'].median())
    df['likes_rolling_mean_5'] = df['likes'].rolling(5, min_periods=1).mean()
    df['likes_rolling_std_5'] = df['likes'].rolling(5, min_periods=1).std().fillna(0)

    # Select features
    baseline_features = [
        'caption_length', 'word_count', 'hashtag_count', 'mention_count',
        'is_video', 'is_weekend'
    ]
    cyclic_features = [
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
    ]
    lag_features = [
        'likes_lag_1', 'likes_lag_2', 'likes_lag_3', 'likes_lag_5',
        'likes_rolling_mean_5', 'likes_rolling_std_5'
    ]
    bert_cols = [c for c in df.columns if c.startswith('bert_')]
    vit_cols = [c for c in df.columns if c.startswith('vit_')]

    all_features = baseline_features + cyclic_features + lag_features + bert_cols + vit_cols

    X = df[all_features]
    y = df['likes'].values

    print(f"  Loaded {len(df)} posts with {X.shape[1]} features")

    return X, y


def objective_rf(trial, X_train, y_train):
    """Optuna objective for Random Forest"""

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 10, 40),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5])
    }

    model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        **params
    )

    # 3-fold CV
    scores = cross_val_score(model, X_train, y_train, cv=3,
                            scoring='neg_mean_absolute_error', n_jobs=-1)

    return -scores.mean()  # Return positive MAE


def objective_hgb(trial, X_train, y_train):
    """Optuna objective for HistGradientBoosting"""

    params = {
        'max_iter': trial.suggest_int('max_iter', 100, 600, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 25),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 0.5)
    }

    model = HistGradientBoostingRegressor(
        random_state=42,
        **params
    )

    # 3-fold CV
    scores = cross_val_score(model, X_train, y_train, cv=3,
                            scoring='neg_mean_absolute_error', n_jobs=-1)

    return -scores.mean()


def objective_gb_meta(trial, X_train, y_train):
    """Optuna objective for GB meta-learner"""

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200, step=25),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0)
    }

    # Note: This is simplified, in practice you'd stack here
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(
        random_state=42,
        **params
    )

    scores = cross_val_score(model, X_train, y_train, cv=3,
                            scoring='neg_mean_absolute_error', n_jobs=-1)

    return -scores.mean()


def run_hyperparam_experiments(n_trials=50):
    """Run hyperparameter search experiments"""

    print("=" * 80)
    print("HYPERPARAMETER SEARCH (Optuna)")
    print("=" * 80)

    # Load data
    X, y = load_full_dataset()

    # Prepare processed data for optimization
    trainer = ModelTrainer(random_state=42)

    # Use 80% for optimization
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    y_train = y[:split_idx]

    # Preprocess
    X_train_proc, y_train_log = trainer.preprocess(
        X_train, y_train,
        pca_components={'bert_': 100, 'vit_': 100},
        fit=True
    )

    # 1. Optimize Random Forest
    print(f"\n[RF] Optimizing Random Forest ({n_trials} trials)...")
    study_rf = optuna.create_study(direction='minimize', study_name='rf_hyperparam')
    study_rf.optimize(
        lambda trial: objective_rf(trial, X_train_proc, y_train_log),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print(f"\n[RF BEST] MAE: {study_rf.best_value:.2f}")
    print(f"  Params: {study_rf.best_params}")

    # 2. Optimize HistGradientBoosting
    print(f"\n[HGB] Optimizing HistGradientBoosting ({n_trials} trials)...")
    study_hgb = optuna.create_study(direction='minimize', study_name='hgb_hyperparam')
    study_hgb.optimize(
        lambda trial: objective_hgb(trial, X_train_proc, y_train_log),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print(f"\n[HGB BEST] MAE: {study_hgb.best_value:.2f}")
    print(f"  Params: {study_hgb.best_params}")

    # 3. Optimize GB meta-learner
    print(f"\n[GB META] Optimizing GB meta-learner ({n_trials} trials)...")
    study_meta = optuna.create_study(direction='minimize', study_name='meta_hyperparam')
    study_meta.optimize(
        lambda trial: objective_gb_meta(trial, X_train_proc, y_train_log),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print(f"\n[META BEST] MAE: {study_meta.best_value:.2f}")
    print(f"  Params: {study_meta.best_params}")

    # 4. Train final model with best params
    print(f"\n[FINAL] Training model with optimized hyperparameters...")

    best_params = {
        'rf_n_estimators': study_rf.best_params['n_estimators'],
        'rf_max_depth': study_rf.best_params['max_depth'],
        'rf_min_samples_split': study_rf.best_params['min_samples_split'],
        'rf_min_samples_leaf': study_rf.best_params['min_samples_leaf'],
        'rf_max_features': study_rf.best_params['max_features'],
        'hgb_max_iter': study_hgb.best_params['max_iter'],
        'hgb_max_depth': study_hgb.best_params['max_depth'],
        'hgb_learning_rate': study_hgb.best_params['learning_rate'],
        'hgb_min_samples_leaf': study_hgb.best_params['min_samples_leaf'],
        'hgb_l2': study_hgb.best_params['l2_regularization'],
        'meta_n_estimators': study_meta.best_params['n_estimators'],
        'meta_max_depth': study_meta.best_params['max_depth'],
        'meta_learning_rate': study_meta.best_params['learning_rate'],
    }

    config = ExperimentConfig(
        name='optimized_hyperparams',
        description=f'Optimized hyperparameters ({n_trials} trials)',
        features=X.columns.tolist(),
        model_type='stacking_gb',
        model_params=best_params,
        preprocessing={
            'pca_components': {'bert_': 100, 'vit_': 100},
            'test_size': 0.2
        },
        tags=['hyperparam_search', 'optimized', 'best']
    )

    result = run_experiment(config=config, X=X, y=y, save_model=True)

    # Save hyperparameter results
    import json
    hyperparam_results = {
        'rf_best': {
            'mae': study_rf.best_value,
            'params': study_rf.best_params
        },
        'hgb_best': {
            'mae': study_hgb.best_value,
            'params': study_hgb.best_params
        },
        'meta_best': {
            'mae': study_meta.best_value,
            'params': study_meta.best_params
        },
        'final_model': {
            'experiment_id': result.experiment_id,
            'metrics': result.metrics
        }
    }

    with open('experiments/hyperparameter_results.json', 'w') as f:
        json.dump(hyperparam_results, f, indent=2)

    print(f"\n[SAVED] Hyperparameter results saved to experiments/hyperparameter_results.json")

    # Print summary
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 80)
    print(f"\nFinal Model Performance:")
    print(f"  Test MAE: {result.metrics['test_mae']:.2f}")
    print(f"  Test R²: {result.metrics['test_r2']:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    args = parser.parse_args()

    run_hyperparam_experiments(n_trials=args.trials)
