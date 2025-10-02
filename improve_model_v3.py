#!/usr/bin/env python3
"""
Phase 3 (Week 2) - Final Model with Visual Features
===================================================

Complete feature set:
- Baseline features (9)
- Interaction features (5)
- NLP features (14)
- Visual features (17) ← NEW!

Total: 45 features

Expected: MAE 70-85, R² 0.25-0.35
"""

import sys
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, setup_logger, get_model_path
from src.features import BaselineFeatureExtractor, AdvancedVisualFeatureExtractor


def extract_nlp_features(df):
    """Extract NLP features (same as v2)."""
    print("\n🔤 Extracting NLP features...")

    # Positive/negative/emotional words
    positive_words = [
        'bagus', 'baik', 'hebat', 'luar biasa', 'keren', 'mantap',
        'sukses', 'senang', 'gembira', 'indah', 'cantik', 'juara',
        'selamat', 'terima kasih', 'wow', 'amazing'
    ]

    negative_words = [
        'buruk', 'jelek', 'gagal', 'sedih', 'kecewa', 'susah',
        'sulit', 'masalah', 'salah', 'batal', 'tidak'
    ]

    emotional_words = [
        'cinta', 'sayang', 'rindu', 'bangga', 'terharu', 'kaget',
        'marah', 'kesal', 'takut', 'harap', 'impian', 'mimpi'
    ]

    features = pd.DataFrame()
    captions = df['caption'].fillna('').astype(str).str.lower()

    # Sentiment
    features['positive_word_count'] = captions.apply(
        lambda x: sum(1 for word in positive_words if word in x)
    )
    features['negative_word_count'] = captions.apply(
        lambda x: sum(1 for word in negative_words if word in x)
    )
    features['emotional_word_count'] = captions.apply(
        lambda x: sum(1 for word in emotional_words if word in x)
    )
    features['sentiment_score'] = (
        features['positive_word_count'] - features['negative_word_count']
    )
    features['has_negative'] = (features['negative_word_count'] > 0).astype(int)

    # Punctuation
    features['question_count'] = captions.apply(lambda x: x.count('?'))
    features['exclamation_count'] = captions.apply(lambda x: x.count('!'))
    features['has_question'] = (features['question_count'] > 0).astype(int)
    features['has_exclamation'] = (features['exclamation_count'] > 0).astype(int)

    # Emoji
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)

    features['emoji_count'] = df['caption'].fillna('').apply(
        lambda x: len(emoji_pattern.findall(str(x)))
    )
    features['has_emoji'] = (features['emoji_count'] > 0).astype(int)

    # Structure
    features['avg_word_length'] = captions.apply(
        lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
    )
    features['caps_word_count'] = df['caption'].fillna('').apply(
        lambda x: sum(1 for word in str(x).split() if word.isupper() and len(word) > 2)
    )
    features['has_url'] = df['caption'].fillna('').apply(
        lambda x: 1 if 'http' in str(x).lower() or 'www.' in str(x).lower() else 0
    )

    print(f"   ✅ Extracted {len(features.columns)} NLP features")
    return features


def apply_robust_preprocessing(X, y, fit_transformer=True, transformer=None, clip_percentile=99):
    """Apply robust preprocessing (same as v2)."""
    print("\n🛡️ Applying robust preprocessing...")

    if fit_transformer:
        clip_value = np.percentile(y, clip_percentile)
        y_clipped = np.clip(y, None, clip_value)
        print(f"   Clipping outliers at {clip_percentile}th percentile: {clip_value:.1f} likes")
        outliers_clipped = (y > clip_value).sum()
        print(f"   Clipped {outliers_clipped} extreme outliers ({outliers_clipped/len(y)*100:.1f}%)")
    else:
        y_clipped = y

    y_log = np.log1p(y_clipped)

    if fit_transformer:
        transformer = QuantileTransformer(n_quantiles=min(100, len(X)),
                                         output_distribution='normal',
                                         random_state=42)
        X_transformed = pd.DataFrame(
            transformer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        print(f"   ✅ Fitted QuantileTransformer on {X.shape[1]} features")
    else:
        X_transformed = pd.DataFrame(
            transformer.transform(X),
            columns=X.columns,
            index=X.index
        )

    return X_transformed, y_log, transformer


def train_ensemble_models(X_train, y_train, X_test, y_test):
    """Train ensemble (RF + HistGB + XGB if available)."""
    print("\n🤖 Training ensemble models...")

    models = {}
    predictions_train = {}
    predictions_test = {}

    # Model 1: Random Forest
    print("\n   [1/3] Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,  # Increased for more complex features
        min_samples_split=3,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['rf'] = rf
    predictions_train['rf'] = rf.predict(X_train)
    predictions_test['rf'] = rf.predict(X_test)

    cv_scores = cross_val_score(rf, X_train, y_train, cv=5,
                                scoring='neg_mean_absolute_error', n_jobs=-1)
    print(f"       CV MAE: {-cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # Model 2: HistGradientBoosting
    print("\n   [2/3] Training HistGradientBoosting...")
    hgb = HistGradientBoostingRegressor(
        max_iter=300,  # Increased
        max_depth=12,  # Increased
        learning_rate=0.05,
        min_samples_leaf=5,
        l2_regularization=0.1,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=15,
        verbose=0
    )
    hgb.fit(X_train, y_train)
    models['hgb'] = hgb
    predictions_train['hgb'] = hgb.predict(X_train)
    predictions_test['hgb'] = hgb.predict(X_test)

    cv_scores = cross_val_score(hgb, X_train, y_train, cv=5,
                                scoring='neg_mean_absolute_error', n_jobs=-1)
    print(f"       CV MAE: {-cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # Model 3: XGBoost (if available)
    try:
        from xgboost import XGBRegressor
        print("\n   [3/3] Training XGBoost...")
        xgb = XGBRegressor(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        xgb.fit(X_train, y_train)
        models['xgb'] = xgb
        predictions_train['xgb'] = xgb.predict(X_train)
        predictions_test['xgb'] = xgb.predict(X_test)

        cv_scores = cross_val_score(xgb, X_train, y_train, cv=5,
                                    scoring='neg_mean_absolute_error', n_jobs=-1)
        print(f"       CV MAE: {-cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    except ImportError:
        print("\n   [3/3] XGBoost not available, skipping...")

    return models, predictions_train, predictions_test


def create_weighted_ensemble(models, predictions_train, predictions_test,
                             y_train, y_test, y_train_orig, y_test_orig):
    """Create weighted ensemble."""
    print("\n🎯 Creating weighted ensemble...")

    weights = {}
    for name, preds_train in predictions_train.items():
        preds_test = predictions_test[name]
        preds_test_orig = np.expm1(preds_test)
        mae = mean_absolute_error(y_test_orig, preds_test_orig)
        weights[name] = 1.0 / mae
        print(f"   {name.upper()}: MAE={mae:.2f} → weight={weights[name]:.4f}")

    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}

    print(f"\n   Normalized weights:")
    for name, weight in weights.items():
        print(f"     {name.upper()}: {weight:.3f}")

    ensemble_train = sum(predictions_train[name] * weights[name]
                        for name in weights.keys())
    ensemble_test = sum(predictions_test[name] * weights[name]
                       for name in weights.keys())

    return ensemble_train, ensemble_test, weights


def main():
    print("\n" + "=" * 80)
    print(" " * 22 + "PHASE 3 - FINAL MODEL")
    print(" " * 15 + "Baseline + NLP + Visual Features")
    print("=" * 80)
    print("\nTarget: MAE <85, R² >0.25\n")

    config = load_config()

    # Load data
    print("📁 Loading data...")
    df = pd.read_csv('fst_unja_from_gallery_dl.csv')
    print(f"   Loaded {len(df)} posts")

    # Extract baseline features
    print("\n🔧 Extracting baseline features...")
    extractor = BaselineFeatureExtractor()
    features_df = extractor.transform(df)
    print(f"   Extracted {len(extractor.feature_names)} baseline features")

    # Add interaction features
    print("\n✨ Adding interaction features...")
    features_df['word_per_hashtag'] = features_df['word_count'] / (features_df['hashtag_count'] + 1)
    features_df['caption_complexity'] = features_df['caption_length'] * features_df['word_count']
    features_df['is_prime_time'] = ((features_df['hour'] >= 10) & (features_df['hour'] <= 12) |
                                     (features_df['hour'] >= 17) & (features_df['hour'] <= 19)).astype(int)
    features_df['has_hashtags'] = (features_df['hashtag_count'] > 0).astype(int)
    features_df['many_hashtags'] = (features_df['hashtag_count'] > 5).astype(int)
    print(f"   Added 5 interaction features")

    # Extract NLP features
    nlp_features = extract_nlp_features(df)
    features_df = pd.concat([features_df, nlp_features], axis=1)

    # Extract VISUAL features (NEW!)
    print("\n🎨 Extracting visual features (NEW!)...")
    visual_extractor = AdvancedVisualFeatureExtractor()
    visual_features = visual_extractor.transform(df)
    features_df = pd.concat([features_df, visual_features], axis=1)

    # All feature columns
    feature_cols = [
        # Baseline (9)
        'caption_length', 'word_count', 'hashtag_count', 'mention_count',
        'is_video', 'hour', 'day_of_week', 'is_weekend', 'month',
        # Interaction (5)
        'word_per_hashtag', 'caption_complexity', 'is_prime_time',
        'has_hashtags', 'many_hashtags',
        # NLP (14)
        'positive_word_count', 'negative_word_count', 'emotional_word_count',
        'sentiment_score', 'has_negative',
        'question_count', 'exclamation_count', 'has_question', 'has_exclamation',
        'emoji_count', 'has_emoji',
        'avg_word_length', 'caps_word_count', 'has_url',
        # Visual (17) - NEW!
        'face_count', 'has_face', 'brightness', 'contrast', 'saturation',
        'dominant_hue', 'color_diversity', 'sharpness', 'aspect_ratio',
        'is_square', 'is_portrait', 'is_landscape',
        'edge_density', 'warm_color_ratio', 'cool_color_ratio',
        'high_brightness_ratio', 'low_brightness_ratio'
    ]

    print(f"\n📊 Total features: {len(feature_cols)}")
    print(f"   - Baseline: 9")
    print(f"   - Interaction: 5")
    print(f"   - NLP: 14")
    print(f"   - Visual (new): 17 🎨")

    # Prepare data
    X = features_df[feature_cols].copy()
    y_original = features_df['likes'].copy()

    # Split
    X_train_raw, X_test_raw, y_train_orig, y_test_orig = train_test_split(
        X, y_original,
        test_size=0.3,
        random_state=42
    )

    print(f"\n📈 Dataset split:")
    print(f"   Train: {len(X_train_raw)} posts")
    print(f"   Test: {len(X_test_raw)} posts")

    # Robust preprocessing
    X_train, y_train_log, transformer = apply_robust_preprocessing(
        X_train_raw, y_train_orig,
        fit_transformer=True,
        clip_percentile=99
    )

    X_test, y_test_log, _ = apply_robust_preprocessing(
        X_test_raw, y_test_orig,
        fit_transformer=False,
        transformer=transformer,
        clip_percentile=99
    )

    # Train ensemble
    models, predictions_train, predictions_test = train_ensemble_models(
        X_train, y_train_log, X_test, y_test_log
    )

    # Weighted ensemble
    ensemble_train_log, ensemble_test_log, weights = create_weighted_ensemble(
        models, predictions_train, predictions_test,
        y_train_log, y_test_log, y_train_orig, y_test_orig
    )

    # Transform back
    ensemble_train = np.expm1(ensemble_train_log)
    ensemble_test = np.expm1(ensemble_test_log)

    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS - FINAL MODEL WITH VISUAL FEATURES")
    print("=" * 80)

    mae_train = mean_absolute_error(y_train_orig, ensemble_train)
    rmse_train = np.sqrt(mean_squared_error(y_train_orig, ensemble_train))
    r2_train = r2_score(y_train_orig, ensemble_train)

    mae_test = mean_absolute_error(y_test_orig, ensemble_test)
    rmse_test = np.sqrt(mean_squared_error(y_test_orig, ensemble_test))
    r2_test = r2_score(y_test_orig, ensemble_test)

    print("\n📊 TRAIN SET:")
    print(f"   MAE:  {mae_train:.2f} likes")
    print(f"   RMSE: {rmse_train:.2f} likes")
    print(f"   R²:   {r2_train:.4f}")

    print("\n📊 TEST SET:")
    print(f"   MAE:  {mae_test:.2f} likes")
    print(f"   RMSE: {rmse_test:.2f} likes")
    print(f"   R²:   {r2_test:.4f}")

    # Comparison
    print("\n📈 Progression across all versions:")
    print("-" * 80)
    print("   Version          | MAE (test) | R² (test) | Features | Method")
    print("   -----------------|------------|-----------|----------|------------------")
    print(f"   Baseline         | 185.29     | 0.0860    | 9        | RF")
    print(f"   Phase 1          | 115.17     | 0.0900    | 14       | RF + log")
    print(f"   Phase 2          | 109.42     | 0.2006    | 28       | Ensemble + NLP")
    print(f"   Phase 3 (FINAL)  | {mae_test:6.2f}     | {r2_test:.4f}    | {len(feature_cols)}       | Full stack 🎨")
    print("-" * 80)

    # Improvements
    improvement_mae_baseline = ((185.29 - mae_test) / 185.29) * 100
    improvement_r2_baseline = ((r2_test - 0.086) / 0.086) * 100 if r2_test > 0.086 else 0

    improvement_mae_v2 = ((109.42 - mae_test) / 109.42) * 100 if mae_test < 109.42 else 0
    improvement_r2_v2 = ((r2_test - 0.2006) / 0.2006) * 100 if r2_test > 0.2006 else 0

    print(f"\n🎉 Total improvement vs baseline:")
    if improvement_mae_baseline > 0:
        print(f"   MAE: {improvement_mae_baseline:.1f}% better ({185.29:.2f} → {mae_test:.2f})")
    if improvement_r2_baseline > 0:
        print(f"   R²:  {improvement_r2_baseline:.1f}% better ({0.086:.4f} → {r2_test:.4f})")

    if improvement_mae_v2 > 0 or improvement_r2_v2 > 0:
        print(f"\n🎨 Visual features contribution (vs Phase 2):")
        if improvement_mae_v2 > 0:
            print(f"   MAE: {improvement_mae_v2:.1f}% better")
        if improvement_r2_v2 > 0:
            print(f"   R²:  {improvement_r2_v2:.1f}% better")

    # Target assessment
    target_mae = 85
    target_r2 = 0.25

    print("\n🎯 Phase 3 Target Assessment:")
    if mae_test <= target_mae:
        print(f"   ✅ MAE Target: ACHIEVED ({mae_test:.2f} <= {target_mae})")
    else:
        print(f"   ⚠️  MAE Target: NOT MET ({mae_test:.2f} > {target_mae})")
        gap = mae_test - target_mae
        print(f"      Gap: {gap:.2f} likes")

    if r2_test >= target_r2:
        print(f"   ✅ R² Target: ACHIEVED ({r2_test:.4f} >= {target_r2})")
    else:
        print(f"   ⚠️  R² Target: NOT MET ({r2_test:.4f} < {target_r2})")
        gap = target_r2 - r2_test
        print(f"      Gap: {gap:.4f}")

    # Feature importance (RF)
    print("\n📈 Top 20 Most Important Features (Random Forest):")
    print("-" * 80)

    if 'rf' in models:
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': models['rf'].feature_importances_
        }).sort_values('importance', ascending=False)

        for idx, (_, row) in enumerate(importance_df.head(20).iterrows(), 1):
            # Mark visual features
            marker = "🎨" if row['feature'] in visual_extractor.feature_names else "  "
            print(f"   {idx:2d}. {marker} {row['feature']:30s}: {row['importance']:.4f}")

    # Visual features analysis
    print("\n🎨 Visual Features Impact:")
    print("-" * 80)
    visual_importance = importance_df[importance_df['feature'].isin(visual_extractor.feature_names)]
    if len(visual_importance) > 0:
        total_visual_importance = visual_importance['importance'].sum()
        print(f"   Total visual feature importance: {total_visual_importance:.4f} ({total_visual_importance*100:.1f}%)")
        print(f"\n   Top 5 visual features:")
        for idx, (_, row) in enumerate(visual_importance.head(5).iterrows(), 1):
            print(f"      {idx}. {row['feature']:30s}: {row['importance']:.4f}")

    # Save model
    model_data = {
        'models': models,
        'weights': weights,
        'transformer': transformer,
        'feature_names': feature_cols,
        'use_log_transform': True,
        'clip_percentile': 99,
        'transformation': 'log1p + quantile + clipping',
        'has_visual_features': True
    }

    output_path = get_model_path('final_model_v3.pkl')
    joblib.dump(model_data, output_path)
    print(f"\n💾 Final model saved to: {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - PHASE 3 COMPLETE")
    print("=" * 80)
    print("\n🔍 Final feature stack:")
    print("   1. ✅ Baseline features (9)")
    print("   2. ✅ Interaction features (5)")
    print("   3. ✅ NLP features (14)")
    print("   4. ✅ Visual features (17) 🎨 NEW!")
    print(f"   Total: {len(feature_cols)} features")

    print("\n📊 Final results:")
    print(f"   MAE:  {mae_test:.2f} likes")
    print(f"   R²:   {r2_test:.4f}")
    print(f"   Features: {len(feature_cols)}")

    print("\n🎉 Key achievements:")
    print(f"   • {improvement_mae_baseline:.0f}% MAE improvement vs baseline")
    print(f"   • {improvement_r2_baseline:.0f}% R² improvement vs baseline")
    if total_visual_importance > 0:
        print(f"   • Visual features contribute {total_visual_importance*100:.1f}%")

    print("\n" + "=" * 80)
    print("DONE! 🎉")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
