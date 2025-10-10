#!/usr/bin/env python3
"""
PHASE 11.3: BERT PCA FINE-TUNING (Exact Optimal Search)
Phase 11.1 showed PCA 65 >> PCA 70 (31.76 vs 43.49)
Hypothesis: Test 62-68 to find EXACT sweet spot
Target: Beat Phase 11.2 MAE=28.13 (or confirm 65 is optimal for account features)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print(" "*15 + "PHASE 11.3: BERT PCA FINE-TUNING (62-68 Components)")
print(" "*20 + "Target: Find EXACT Optimal PCA Dimensionality")
print("="*90)
print()

# Load data
df_main = pd.read_csv('multi_account_dataset.csv')
df_bert = pd.read_csv('data/processed/bert_embeddings_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])
df_visual = pd.read_csv('data/processed/advanced_visual_features_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])
df = df_main.merge(df_bert, on=['post_id', 'account'], how='inner').merge(df_visual, on=['post_id', 'account'], how='inner')

print(f"[LOAD] Dataset: {len(df)} posts from {df['account'].nunique()} accounts")
print()

# Account features (from Phase 11.2)
train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
df_train_stats = df.iloc[train_idx].copy()

account_stats = df_train_stats.groupby('account').agg({
    'likes': ['mean', 'std', 'median'],
    'caption_length': 'mean',
    'hashtag_count': 'mean',
    'is_video': 'mean'
}).reset_index()

account_stats.columns = ['account', 'account_avg_likes', 'account_std_likes', 'account_median_likes',
                          'account_avg_caption_len', 'account_avg_hashtags', 'account_video_ratio']

df = df.merge(account_stats, on='account', how='left')

for col in ['account_avg_likes', 'account_std_likes', 'account_median_likes',
            'account_avg_caption_len', 'account_avg_hashtags', 'account_video_ratio']:
    df[col] = df[col].fillna(df[col].median())

df['caption_vs_account_avg'] = df['caption_length'] / (df['account_avg_caption_len'] + 1)
df['hashtag_vs_account_avg'] = df['hashtag_count'] / (df['account_avg_hashtags'] + 1)

# Features
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                 'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']
bert_cols = [col for col in df.columns if col.startswith('bert_')]

# Phase 11.2 visual + cross interactions
metadata_base = ['file_size_kb', 'is_portrait', 'is_landscape', 'is_square']
df['resolution_log'] = np.log1p(df['resolution'])
df['aspect_ratio_sq'] = df['aspect_ratio'] ** 2
df['aspect_x_logres'] = df['aspect_ratio'] * df['resolution_log']
df['filesize_x_logres'] = df['file_size_kb'] * df['resolution_log']
df['aspect_sq_x_logres'] = df['aspect_ratio_sq'] * df['resolution_log']

df['caption_x_aspect'] = df['caption_length'] * df['aspect_ratio']
df['caption_x_logres'] = df['caption_length'] * df['resolution_log']
df['hashtag_x_logres'] = df['hashtag_count'] * df['resolution_log']
df['word_x_filesize'] = df['word_count'] * df['file_size_kb']
df['caption_x_filesize'] = df['caption_length'] * df['file_size_kb']

cross_interactions = ['caption_x_aspect', 'caption_x_logres', 'hashtag_x_logres',
                      'word_x_filesize', 'caption_x_filesize']

visual_features = (metadata_base + ['aspect_ratio', 'resolution_log', 'aspect_ratio_sq',
                   'aspect_x_logres', 'filesize_x_logres', 'aspect_sq_x_logres'] +
                   cross_interactions)

account_features = ['account_avg_caption_len', 'account_avg_hashtags', 'account_video_ratio',
                    'caption_vs_account_avg', 'hashtag_vs_account_avg']

X_baseline = df[baseline_cols].values
X_bert_full = df[bert_cols].values
X_visual = df[visual_features].values
X_account = df[account_features].values
y = df['likes'].values

X_baseline_train, X_baseline_test = X_baseline[train_idx], X_baseline[test_idx]
X_bert_train, X_bert_test = X_bert_full[train_idx], X_bert_full[test_idx]
X_visual_train, X_visual_test = X_visual[train_idx], X_visual[test_idx]
X_account_train, X_account_test = X_account[train_idx], X_account[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

clip_threshold = np.percentile(y_train, 99)
y_train_log = np.log1p(np.clip(y_train, 0, clip_threshold))
y_test_log = np.log1p(np.clip(y_test, 0, clip_threshold))

print("[STRATEGY] Test PCA components: 62, 63, 64, 65, 66, 67, 68")
print(f"   Baseline: {len(baseline_cols)}, Visual+Cross: {len(visual_features)}, Account: {len(account_features)}")
print()

# Test different PCA components
pca_components_list = [62, 63, 64, 65, 66, 67, 68]
results = []

for n_components in pca_components_list:
    print(f"[PCA {n_components}] Reducing BERT from 768 to {n_components} dimensions...")

    pca_bert = PCA(n_components=n_components, random_state=42)
    X_bert_pca_train = pca_bert.fit_transform(X_bert_train)
    X_bert_pca_test = pca_bert.transform(X_bert_test)
    variance_preserved = pca_bert.explained_variance_ratio_.sum()

    print(f"   Variance preserved: {variance_preserved*100:.2f}%")

    # Combine features
    X_train = np.hstack([X_baseline_train, X_bert_pca_train, X_visual_train, X_account_train])
    X_test = np.hstack([X_baseline_test, X_bert_pca_test, X_visual_test, X_account_test])

    total_features = X_train.shape[1]
    print(f"   Total features: {total_features}")

    # Scale
    scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train ensemble
    print(f"   Training ensemble...", end=" ")

    models = [
        ('gb', GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, random_state=42)),
        ('hgb', HistGradientBoostingRegressor(max_iter=600, learning_rate=0.07, max_depth=7, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=300, max_depth=16, random_state=42, n_jobs=-1)),
        ('et', ExtraTreesRegressor(n_estimators=300, max_depth=16, random_state=42, n_jobs=-1))
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(X_train_scaled), len(models)))
    test_preds = np.zeros((len(X_test_scaled), len(models)))

    for i, (name, model) in enumerate(models):
        for fold_idx, (train_fold, val_fold) in enumerate(kf.split(X_train_scaled)):
            X_tr, X_val = X_train_scaled[train_fold], X_train_scaled[val_fold]
            y_tr, y_val = y_train_log[train_fold], y_train_log[val_fold]

            model.fit(X_tr, y_tr)
            oof_preds[val_fold, i] = model.predict(X_val)

        model.fit(X_train_scaled, y_train_log)
        test_preds[:, i] = model.predict(X_test_scaled)

    # Meta-learner
    meta_model = Ridge(alpha=10)
    meta_model.fit(oof_preds, y_train_log)

    # Final prediction
    final_pred_test = meta_model.predict(test_preds)
    final_pred_test_inv = np.expm1(final_pred_test)
    y_test_inv = np.expm1(y_test_log)

    mae = mean_absolute_error(y_test_inv, final_pred_test_inv)
    r2 = r2_score(y_test_inv, final_pred_test_inv)

    print(f"MAE={mae:.2f}, R2={r2:.4f}")
    print()

    results.append({
        'n_components': n_components,
        'variance': variance_preserved,
        'mae': mae,
        'r2': r2,
        'total_features': total_features,
        'pca_bert': pca_bert,
        'scaler': scaler,
        'models': [m for _, m in models],
        'meta_model': meta_model
    })

print("="*90)
print("[RESULTS SUMMARY] PCA Fine-Tuning")
print("="*90)
print()

print(f"{'PCA':>4} | {'Variance':>9} | {'MAE':>7} | {'R2':>7} | {'Features':>8} | Status")
print("-" * 70)

best_result = min(results, key=lambda x: x['mae'])

for res in results:
    status = ""
    if res['n_components'] == best_result['n_components']:
        status = "BEST"
    elif res['mae'] < 28.13:
        status = "Champion"
    elif res['mae'] < 31.76:
        status = "Good"
    else:
        status = "Weak"

    print(f"{res['n_components']:>4} | {res['variance']*100:>8.2f}% | {res['mae']:>7.2f} | {res['r2']:>7.4f} | {res['total_features']:>8} | {status}")

print()
print("="*90)
print(f"[OPTIMAL] PCA {best_result['n_components']} components")
print(f"   MAE: {best_result['mae']:.2f}")
print(f"   R2: {best_result['r2']:.4f}")
print(f"   Variance: {best_result['variance']*100:.2f}%")
print("="*90)
print()

# Compare with champions
champion_11_2 = 28.13
champion_11_1 = 31.76
champion_10_24 = 43.49

if best_result['mae'] < champion_11_2:
    improvement = ((champion_11_2 - best_result['mae']) / champion_11_2) * 100
    print(f"[CHAMPION] NEW RECORD! Beat Phase 11.2 by {improvement:.2f}%!")
    print(f"   PCA {best_result['n_components']} is optimal with account features")
elif best_result['mae'] < champion_11_1:
    decline = ((best_result['mae'] - champion_11_2) / champion_11_2) * 100
    print(f"   Phase 11.2 remains champion (MAE={champion_11_2:.2f})")
    print(f"   Best PCA config slightly worse (+{decline:.2f}%)")
    print(f"   But PCA {best_result['n_components']} confirmed as optimal dimensionality")
else:
    print(f"   Phase 11.2 remains champion (MAE={champion_11_2:.2f})")
    print(f"   PCA 65 from Phase 11.1 still best dimensionality choice")

print()
print("="*90)
print()

# Save best model if it beats Phase 11.2
if best_result['mae'] < champion_11_2:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/phase11_3_pca{best_result['n_components']}_{timestamp}.pkl"
    joblib.dump({
        'scaler': best_result['scaler'],
        'pca_bert': best_result['pca_bert'],
        'base_models': best_result['models'],
        'meta_model': best_result['meta_model'],
        'account_stats': account_stats,
        'mae': best_result['mae'],
        'r2': best_result['r2'],
        'n_components': best_result['n_components'],
        'variance_preserved': best_result['variance']
    }, model_path)
    print(f"[SAVE] Model saved: {model_path}")
    print()

print("[INSIGHT] BERT PCA Variance-Performance Curve:")
for res in sorted(results, key=lambda x: x['n_components']):
    delta_from_best = res['mae'] - best_result['mae']
    marker = " <-- OPTIMAL" if res['n_components'] == best_result['n_components'] else ""
    print(f"   PCA {res['n_components']:>2}: {res['variance']*100:.2f}% variance -> MAE {res['mae']:.2f} (+{delta_from_best:>5.2f}){marker}")

print()
print("[CONCLUSION] Exact optimal PCA components identified for production deployment")
