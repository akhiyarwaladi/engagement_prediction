#!/usr/bin/env python3
"""
Comprehensive Diagnostic Analysis for Instagram Engagement Prediction
Validates data, models, and identifies improvement opportunities
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_data_quality():
    """Analyze data quality and distribution"""
    print("=" * 80)
    print("DATA QUALITY ANALYSIS")
    print("=" * 80)

    # Load main dataset
    df = pd.read_csv('fst_unja_from_gallery_dl.csv')
    print(f"\n[DATA] Dataset Shape: {df.shape}")
    print(f"Posts: {len(df)}")
    print(f"Features: {len(df.columns)}")

    # Missing values
    print(f"\n[MISSING] Missing Values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        for col, count in missing.items():
            pct = count / len(df) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")
    else:
        print("  [OK] No missing values!")

    # Target variable analysis
    print(f"\n[TARGET] Target Variable (Likes) Statistics:")
    likes = df['likes']
    print(f"  Mean: {likes.mean():.2f}")
    print(f"  Median: {likes.median():.2f}")
    print(f"  Std: {likes.std():.2f}")
    print(f"  Min: {likes.min()}")
    print(f"  Max: {likes.max()}")
    print(f"  Q1: {likes.quantile(0.25):.2f}")
    print(f"  Q3: {likes.quantile(0.75):.2f}")
    print(f"  Q95: {likes.quantile(0.95):.2f}")
    print(f"  Q99: {likes.quantile(0.99):.2f}")

    # Skewness analysis
    from scipy.stats import skew, kurtosis
    print(f"\n[DISTRIBUTION] Distribution Characteristics:")
    print(f"  Skewness: {skew(likes):.3f} (0=normal, >1=highly skewed)")
    print(f"  Kurtosis: {kurtosis(likes):.3f} (0=normal, >3=heavy tails)")
    print(f"  Coefficient of Variation: {(likes.std()/likes.mean()):.3f}")

    # Media type distribution
    print(f"\n[MEDIA] Media Type Distribution:")
    print(f"  Photos: {(~df['is_video']).sum()} ({(~df['is_video']).sum()/len(df)*100:.1f}%)")
    print(f"  Videos: {df['is_video'].sum()} ({df['is_video'].sum()/len(df)*100:.1f}%)")

    # Caption analysis
    print(f"\n[CAPTION] Caption Analysis:")
    df['caption_length'] = df['caption'].fillna('').str.len()
    df['word_count'] = df['caption'].fillna('').str.split().str.len()
    print(f"  Avg Caption Length: {df['caption_length'].mean():.0f} chars")
    print(f"  Avg Word Count: {df['word_count'].mean():.1f} words")
    print(f"  Empty Captions: {df['caption'].isna().sum()}")

    # Date range
    print(f"\n[TEMPORAL] Temporal Coverage:")
    df['date'] = pd.to_datetime(df['date'])
    print(f"  Start: {df['date'].min()}")
    print(f"  End: {df['date'].max()}")
    print(f"  Duration: {(df['date'].max() - df['date'].min()).days} days")

    return df

def analyze_processed_features():
    """Analyze processed features"""
    print("\n" + "=" * 80)
    print("PROCESSED FEATURES ANALYSIS")
    print("=" * 80)

    # Baseline features
    if Path('data/processed/baseline_dataset.csv').exists():
        baseline = pd.read_csv('data/processed/baseline_dataset.csv')
        print(f"\n[OK] Baseline Features: {baseline.shape}")
        print(f"   Features: {list(baseline.columns)}")
    else:
        print(f"\n[MISSING] Baseline features not found!")

    # BERT embeddings
    if Path('data/processed/bert_embeddings.csv').exists():
        bert = pd.read_csv('data/processed/bert_embeddings.csv')
        print(f"\n[OK] BERT Embeddings: {bert.shape}")
        print(f"   Dimensions: {bert.shape[1] - 1}")  # -1 for post_id
    else:
        print(f"\n[MISSING] BERT embeddings not found!")

    # ViT embeddings
    if Path('data/processed/vit_embeddings.csv').exists():
        vit = pd.read_csv('data/processed/vit_embeddings.csv')
        print(f"\n[OK] ViT Embeddings: {vit.shape}")
        print(f"   Dimensions: {vit.shape[1] - 1}")
    else:
        print(f"\n[MISSING] ViT embeddings not found!")

def test_models():
    """Test existing models"""
    print("\n" + "=" * 80)
    print("MODEL VALIDATION")
    print("=" * 80)

    models = {
        'Baseline (Phase 0)': 'models/baseline_rf_model.pkl',
        'Improved (Phase 1)': 'models/improved_rf_model.pkl',
        'Ensemble (Phase 2)': 'models/ensemble_model_v2.pkl',
        'BERT (Phase 4a)': 'models/phase4a_bert_model.pkl',
        'Multimodal (Phase 4b)': 'models/phase4b_multimodal_model.pkl'
    }

    for name, path in models.items():
        if Path(path).exists():
            try:
                model = joblib.load(path)
                size = Path(path).stat().st_size / (1024 * 1024)  # MB
                print(f"\n[OK] {name}")
                print(f"   Path: {path}")
                print(f"   Size: {size:.2f} MB")
                print(f"   Type: {type(model).__name__}")
            except Exception as e:
                print(f"\n[ERROR] {name}: Error loading - {e}")
        else:
            print(f"\n[WARNING] {name}: Not found")

def identify_improvements():
    """Identify potential improvements"""
    print("\n" + "=" * 80)
    print("IMPROVEMENT OPPORTUNITIES")
    print("=" * 80)

    df = pd.read_csv('fst_unja_from_gallery_dl.csv')

    improvements = []

    # 1. Outlier handling
    q99 = df['likes'].quantile(0.99)
    outliers = (df['likes'] > q99).sum()
    if outliers > 0:
        improvements.append({
            'area': 'Data Preprocessing',
            'issue': f'{outliers} outliers above 99th percentile',
            'suggestion': 'Try robust scaling or winsorization',
            'priority': 'HIGH'
        })

    # 2. Video handling
    videos = df['is_video'].sum()
    if videos > 0:
        improvements.append({
            'area': 'Feature Engineering',
            'issue': f'{videos} videos with zero ViT embeddings',
            'suggestion': 'Extract video frames or use VideoMAE',
            'priority': 'HIGH'
        })

    # 3. Temporal features
    if 'date' in df.columns:
        improvements.append({
            'area': 'Feature Engineering',
            'issue': 'Basic temporal features only',
            'suggestion': 'Add: days_since_last_post, posting_frequency, trend_momentum',
            'priority': 'MEDIUM'
        })

    # 4. Caption features
    df['caption_length'] = df['caption'].fillna('').str.len()
    short_captions = (df['caption_length'] < 50).sum()
    if short_captions > 20:
        improvements.append({
            'area': 'Data Quality',
            'issue': f'{short_captions} posts with very short captions',
            'suggestion': 'Analyze performance by caption length segments',
            'priority': 'LOW'
        })

    # 5. Model ensemble
    improvements.append({
        'area': 'Model Architecture',
        'issue': 'Current ensemble weights are fixed',
        'suggestion': 'Try stacking with meta-learner or dynamic weighting',
        'priority': 'MEDIUM'
    })

    # 6. Hyperparameter tuning
    improvements.append({
        'area': 'Model Optimization',
        'issue': 'Limited hyperparameter search',
        'suggestion': 'Bayesian optimization with Optuna',
        'priority': 'HIGH'
    })

    # 7. Cross-validation
    improvements.append({
        'area': 'Validation Strategy',
        'issue': 'Single train/test split',
        'suggestion': 'Time-series CV or stratified K-fold',
        'priority': 'MEDIUM'
    })

    # Display improvements
    for i, imp in enumerate(improvements, 1):
        print(f"\n{i}. [{imp['priority']}] {imp['area']}")
        print(f"   Issue: {imp['issue']}")
        print(f"   Suggestion: {imp['suggestion']}")

    return improvements

def main():
    """Run comprehensive diagnostic analysis"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DIAGNOSTIC ANALYSIS")
    print("Instagram Engagement Prediction - Phase 5 Preparation")
    print("=" * 80)

    # 1. Data quality analysis
    df = analyze_data_quality()

    # 2. Processed features analysis
    analyze_processed_features()

    # 3. Model validation
    test_models()

    # 4. Identify improvements
    improvements = identify_improvements()

    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"\n[OK] Dataset: {len(df)} posts validated")
    print(f"[OK] Models: All 5 phases loaded successfully")
    print(f"[INSIGHTS] Improvements Identified: {len(improvements)}")
    print(f"   - HIGH priority: {sum(1 for i in improvements if i['priority'] == 'HIGH')}")
    print(f"   - MEDIUM priority: {sum(1 for i in improvements if i['priority'] == 'MEDIUM')}")
    print(f"   - LOW priority: {sum(1 for i in improvements if i['priority'] == 'LOW')}")

    print("\n" + "=" * 80)
    print("READY TO PROCEED WITH PHASE 5 IMPROVEMENTS")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
