#!/usr/bin/env python3
"""
ABLATION STUDY: Test impact of each feature group
Run multiple experiments to understand what works
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from src.training.experiment import ExperimentConfig, get_tracker
from src.training.trainer import run_experiment


def load_data():
    """Load dataset and features"""
    print("\n[DATA] Loading dataset...")

    # Load main dataset
    df = pd.read_csv('fst_unja_from_gallery_dl.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Load processed embeddings
    bert_df = pd.read_csv('data/processed/bert_embeddings.csv')
    vit_df = pd.read_csv('data/processed/vit_embeddings_enhanced.csv')

    # Merge
    df = df.merge(bert_df, on='post_id', how='left')
    df = df.merge(vit_df, on='post_id', how='left')

    print(f"  Loaded {len(df)} posts")

    return df


def create_baseline_features(df):
    """Create baseline features"""
    df = df.copy()

    df['caption'] = df['caption'].fillna('')
    df['caption_length'] = df['caption'].str.len()
    df['word_count'] = df['caption'].str.split().str.len()
    df['hashtag_count'] = df['hashtags_count']
    df['mention_count'] = df['mentions_count']
    df['is_video'] = df['is_video'].astype(int)
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)

    return df[['caption_length', 'word_count', 'hashtag_count', 'mention_count',
               'is_video', 'is_weekend']]


def create_cyclic_features(df):
    """Create cyclic temporal features"""
    df = df.copy()

    hour = df['date'].dt.hour
    day = df['date'].dt.dayofweek
    month = df['date'].dt.month

    return pd.DataFrame({
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'day_sin': np.sin(2 * np.pi * day / 7),
        'day_cos': np.cos(2 * np.pi * day / 7),
        'month_sin': np.sin(2 * np.pi * (month - 1) / 12),
        'month_cos': np.cos(2 * np.pi * (month - 1) / 12),
    })


def create_lag_features(df):
    """Create engagement lag features"""
    df = df.copy()

    lag_df = pd.DataFrame()
    for lag in [1, 2, 3, 5]:
        col = f'likes_lag_{lag}'
        lag_df[col] = df['likes'].shift(lag).fillna(df['likes'].median())

    lag_df['likes_rolling_mean_5'] = df['likes'].rolling(5, min_periods=1).mean()
    lag_df['likes_rolling_std_5'] = df['likes'].rolling(5, min_periods=1).std().fillna(0)

    return lag_df


def run_ablation_experiments():
    """Run ablation study - test each feature group"""

    print("=" * 80)
    print("ABLATION STUDY: Testing Feature Importance")
    print("=" * 80)

    # Load data
    df = load_data()
    y = df['likes'].values

    # Prepare feature groups
    print("\n[FEATURES] Creating feature groups...")

    baseline = create_baseline_features(df)
    cyclic = create_cyclic_features(df)
    lag = create_lag_features(df)

    bert_cols = [c for c in df.columns if c.startswith('bert_')]
    vit_cols = [c for c in df.columns if c.startswith('vit_')]

    bert = df[bert_cols]
    vit = df[vit_cols]

    print(f"  Baseline: {baseline.shape[1]} features")
    print(f"  Cyclic: {cyclic.shape[1]} features")
    print(f"  Lag: {lag.shape[1]} features")
    print(f"  BERT: {bert.shape[1]} features")
    print(f"  ViT: {vit.shape[1]} features")

    # Define experiments
    experiments = [
        # 1. Baseline only
        {
            'name': 'baseline_only',
            'description': 'Baseline features only',
            'features': baseline,
            'pca': None,
            'tags': ['ablation', 'baseline']
        },

        # 2. Baseline + Cyclic
        {
            'name': 'baseline_cyclic',
            'description': 'Baseline + Cyclic temporal',
            'features': pd.concat([baseline, cyclic], axis=1),
            'pca': None,
            'tags': ['ablation', 'temporal']
        },

        # 3. Baseline + Cyclic + Lag
        {
            'name': 'baseline_cyclic_lag',
            'description': 'Baseline + Cyclic + Lag',
            'features': pd.concat([baseline, cyclic, lag], axis=1),
            'pca': None,
            'tags': ['ablation', 'temporal', 'lag']
        },

        # 4. Baseline + BERT (no PCA)
        {
            'name': 'baseline_bert_nopca',
            'description': 'Baseline + BERT (768-dim, no PCA)',
            'features': pd.concat([baseline, bert], axis=1),
            'pca': None,
            'tags': ['ablation', 'bert', 'nopca']
        },

        # 5. Baseline + BERT (PCA 50)
        {
            'name': 'baseline_bert_pca50',
            'description': 'Baseline + BERT (PCA 50)',
            'features': pd.concat([baseline, bert], axis=1),
            'pca': {'bert_': 50},
            'tags': ['ablation', 'bert', 'pca50']
        },

        # 6. Baseline + BERT (PCA 100)
        {
            'name': 'baseline_bert_pca100',
            'description': 'Baseline + BERT (PCA 100)',
            'features': pd.concat([baseline, bert], axis=1),
            'pca': {'bert_': 100},
            'tags': ['ablation', 'bert', 'pca100']
        },

        # 7. Baseline + ViT (PCA 50)
        {
            'name': 'baseline_vit_pca50',
            'description': 'Baseline + ViT (PCA 50)',
            'features': pd.concat([baseline, vit], axis=1),
            'pca': {'vit_': 50},
            'tags': ['ablation', 'vit', 'pca50']
        },

        # 8. Baseline + Cyclic + Lag + BERT (PCA 100)
        {
            'name': 'temporal_bert',
            'description': 'Temporal + BERT',
            'features': pd.concat([baseline, cyclic, lag, bert], axis=1),
            'pca': {'bert_': 100},
            'tags': ['ablation', 'temporal', 'bert']
        },

        # 9. Baseline + Cyclic + Lag + ViT (PCA 100)
        {
            'name': 'temporal_vit',
            'description': 'Temporal + ViT',
            'features': pd.concat([baseline, cyclic, lag, vit], axis=1),
            'pca': {'vit_': 100},
            'tags': ['ablation', 'temporal', 'vit']
        },

        # 10. FULL MODEL (Phase 5.1 recreation)
        {
            'name': 'full_model',
            'description': 'All features (Baseline + Temporal + BERT + ViT)',
            'features': pd.concat([baseline, cyclic, lag, bert, vit], axis=1),
            'pca': {'bert_': 100, 'vit_': 100},
            'tags': ['ablation', 'full', 'best']
        },
    ]

    # Run experiments
    print(f"\n[EXPERIMENTS] Running {len(experiments)} experiments...")

    results = []

    for i, exp in enumerate(experiments, 1):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i}/{len(experiments)}: {exp['name']}")
        print(f"{'='*80}")

        config = ExperimentConfig(
            name=exp['name'],
            description=exp['description'],
            features=exp['features'].columns.tolist(),
            model_type='stacking_gb',  # Use GradientBoosting meta-learner
            model_params={
                'rf_n_estimators': 300,
                'rf_max_depth': 26,
                'hgb_max_iter': 254,
                'hgb_max_depth': 15,
                'hgb_learning_rate': 0.104,
                'meta_n_estimators': 100,
                'meta_max_depth': 5,
                'meta_learning_rate': 0.05
            },
            preprocessing={
                'pca_components': exp['pca'],
                'test_size': 0.2
            },
            tags=exp['tags']
        )

        result = run_experiment(
            config=config,
            X=exp['features'],
            y=y,
            save_model=True
        )

        results.append(result)

    # Print summary
    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETE")
    print("=" * 80)

    tracker = get_tracker()
    tracker.print_summary()


if __name__ == "__main__":
    run_ablation_experiments()
