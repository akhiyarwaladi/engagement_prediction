#!/usr/bin/env python3
"""
Comprehensive comparison of all phases
Analyze performance evolution and identify key insights
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_phase_results():
    """Load documented results from all phases"""

    results = {
        'Phase 0 (Baseline)': {
            'MAE': 185.29,
            'R2': 0.086,
            'Features': 9,
            'Method': 'Random Forest',
            'Key_Innovation': 'Baseline features only'
        },
        'Phase 1 (Log+Interactions)': {
            'MAE': 115.17,
            'R2': 0.090,
            'Features': 14,
            'Method': 'RF + log transform',
            'Key_Innovation': 'Log transform + feature interactions'
        },
        'Phase 2 (NLP+Ensemble)': {
            'MAE': 109.42,
            'R2': 0.200,
            'Features': 28,
            'Method': 'Ensemble + NLP',
            'Key_Innovation': 'TF-IDF + sentiment + ensemble'
        },
        'Phase 4a (IndoBERT)': {
            'MAE': 98.94,
            'R2': 0.206,
            'Features': 59,
            'Method': 'RF + IndoBERT',
            'Key_Innovation': 'Indonesian BERT embeddings'
        },
        'Phase 4b (Multimodal)': {
            'MAE': 111.28,
            'R2': 0.234,
            'Features': 109,
            'Method': 'Ensemble + BERT + ViT',
            'Key_Innovation': 'Multimodal (text + vision)'
        },
        'Phase 5 (Ultra-Optimized)': {
            'MAE': 88.28,
            'R2': 0.483,
            'Features': 114,
            'Method': 'Stacking + Optuna + Video',
            'Key_Innovation': 'Video frames + temporal + optimization'
        }
    }

    return pd.DataFrame(results).T

def analyze_improvements():
    """Analyze improvements between phases"""
    print("=" * 80)
    print("COMPREHENSIVE PHASE COMPARISON")
    print("=" * 80)

    df = load_phase_results()

    # Display results
    print("\n[RESULTS] All Phases Performance:")
    print(df.to_string())

    # Calculate improvements
    print("\n" + "=" * 80)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 80)

    baseline_mae = df.loc['Phase 0 (Baseline)', 'MAE']
    baseline_r2 = df.loc['Phase 0 (Baseline)', 'R2']

    for phase in df.index:
        mae = df.loc[phase, 'MAE']
        r2 = df.loc[phase, 'R2']

        mae_improvement = ((baseline_mae - mae) / baseline_mae) * 100
        r2_improvement = ((r2 - baseline_r2) / baseline_r2) * 100

        print(f"\n{phase}:")
        print(f"  MAE: {mae:.2f} ({mae_improvement:+.1f}% vs baseline)")
        print(f"  R²: {r2:.3f} ({r2_improvement:+.1f}% vs baseline)")
        print(f"  Innovation: {df.loc[phase, 'Key_Innovation']}")

    # Best performers
    print("\n" + "=" * 80)
    print("BEST PERFORMERS")
    print("=" * 80)

    best_mae = df['MAE'].idxmin()
    best_r2 = df['R2'].idxmax()

    print(f"\n[BEST MAE] {best_mae}")
    print(f"  MAE: {df.loc[best_mae, 'MAE']:.2f} likes")
    print(f"  Method: {df.loc[best_mae, 'Method']}")

    print(f"\n[BEST R²] {best_r2}")
    print(f"  R²: {df.loc[best_r2, 'R2']:.3f}")
    print(f"  Method: {df.loc[best_r2, 'Method']}")

    # Phase 5 specific analysis
    print("\n" + "=" * 80)
    print("PHASE 5 BREAKTHROUGH ANALYSIS")
    print("=" * 80)

    phase5_mae = df.loc['Phase 5 (Ultra-Optimized)', 'MAE']
    phase5_r2 = df.loc['Phase 5 (Ultra-Optimized)', 'R2']
    phase4a_mae = df.loc['Phase 4a (IndoBERT)', 'MAE']
    phase4b_r2 = df.loc['Phase 4b (Multimodal)', 'R2']

    print(f"\nPhase 5 vs Phase 4a (best MAE):")
    print(f"  MAE: {phase5_mae:.2f} vs {phase4a_mae:.2f}")
    print(f"  Improvement: {phase4a_mae - phase5_mae:.2f} likes ({((phase4a_mae - phase5_mae)/phase4a_mae)*100:.1f}%)")

    print(f"\nPhase 5 vs Phase 4b (best R² before Phase 5):")
    print(f"  R²: {phase5_r2:.3f} vs {phase4b_r2:.3f}")
    print(f"  Improvement: {phase5_r2 - phase4b_r2:.3f} ({((phase5_r2 - phase4b_r2)/phase4b_r2)*100:.1f}%)")

    print(f"\nPhase 5 vs Baseline:")
    print(f"  MAE reduction: {baseline_mae - phase5_mae:.2f} likes ({((baseline_mae - phase5_mae)/baseline_mae)*100:.1f}%)")
    print(f"  R² improvement: {phase5_r2 - baseline_r2:.3f} ({((phase5_r2 - baseline_r2)/baseline_r2)*100:.1f}%)")

    return df

def identify_next_improvements():
    """Identify opportunities for Phase 5.1"""
    print("\n" + "=" * 80)
    print("NEXT IMPROVEMENT OPPORTUNITIES (Phase 5.1)")
    print("=" * 80)

    improvements = [
        {
            'id': 1,
            'area': 'Ensemble Weighting',
            'current': 'Fixed stacking with Ridge meta-learner',
            'improvement': 'Try Gradient Boosting or Neural Network as meta-learner',
            'expected_gain': 'MAE -3 to -5',
            'priority': 'HIGH'
        },
        {
            'id': 2,
            'area': 'Temporal Features',
            'current': '5 temporal features',
            'improvement': 'Add: hour_sine/cosine, day_sine/cosine, engagement_lag_features',
            'expected_gain': 'R² +0.02 to +0.05',
            'priority': 'HIGH'
        },
        {
            'id': 3,
            'area': 'Video Embeddings',
            'current': '3 frames average',
            'improvement': 'Try 5-7 frames or temporal attention pooling',
            'expected_gain': 'MAE -2 to -3 for videos',
            'priority': 'MEDIUM'
        },
        {
            'id': 4,
            'area': 'PCA Components',
            'current': 'BERT: 50, ViT: 50',
            'improvement': 'Increase to BERT: 100, ViT: 100 (preserve 98%+ variance)',
            'expected_gain': 'R² +0.01 to +0.03',
            'priority': 'MEDIUM'
        },
        {
            'id': 5,
            'area': 'Cross-Validation',
            'current': 'Single train/test split',
            'improvement': 'Time-series cross-validation with 5 folds',
            'expected_gain': 'More robust estimates',
            'priority': 'HIGH'
        },
        {
            'id': 6,
            'area': 'Optuna Trials',
            'current': '30 trials',
            'improvement': 'Increase to 100 trials with pruning',
            'expected_gain': 'MAE -1 to -2',
            'priority': 'MEDIUM'
        },
        {
            'id': 7,
            'area': 'Feature Interactions',
            'current': 'No explicit interactions',
            'improvement': 'Add: temporal × BERT, temporal × ViT interactions',
            'expected_gain': 'R² +0.02 to +0.04',
            'priority': 'HIGH'
        },
        {
            'id': 8,
            'area': 'Caption Preprocessing',
            'current': 'Raw captions to BERT',
            'improvement': 'Clean emojis, normalize mentions/hashtags',
            'expected_gain': 'MAE -1 to -2',
            'priority': 'LOW'
        }
    ]

    for imp in improvements:
        print(f"\n{imp['id']}. [{imp['priority']}] {imp['area']}")
        print(f"   Current: {imp['current']}")
        print(f"   Improvement: {imp['improvement']}")
        print(f"   Expected Gain: {imp['expected_gain']}")

    # Prioritize
    high_priority = [i for i in improvements if i['priority'] == 'HIGH']

    print("\n" + "=" * 80)
    print("RECOMMENDED PHASE 5.1 IMPLEMENTATION")
    print("=" * 80)

    print("\nImplement these HIGH priority improvements:")
    for i, imp in enumerate(high_priority, 1):
        print(f"  {i}. {imp['area']}: {imp['improvement']}")

    print(f"\nExpected Phase 5.1 Performance:")
    print(f"  MAE: 78-83 likes (current: 88.28)")
    print(f"  R²: 0.52-0.56 (current: 0.483)")

    return improvements

def main():
    """Main execution"""

    # Analyze all phases
    df = analyze_improvements()

    # Identify next improvements
    improvements = identify_next_improvements()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n[SUCCESS] Phase 5 achieved:")
    print("  - Best MAE ever: 88.28 likes")
    print("  - Best R² ever: 0.483")
    print("  - 52.4% MAE reduction from baseline")
    print("  - 461.6% R² improvement from baseline")

    print("\n[NEXT] Phase 5.1 targets:")
    print("  - MAE: < 83 likes (5+ likes improvement)")
    print("  - R²: > 0.52 (0.037+ improvement)")
    print("  - Focus: Ensemble meta-learner + temporal features + interactions")

    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()
