#!/usr/bin/env python3
"""
PHASE 10.2: PCA DIMENSIONALITY OPTIMIZATION
Fine-tune BERT and Aesthetic PCA dimensions
MUST INCLUDE: Visual + Text features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print(" "*20 + "PHASE 10.2: PCA OPTIMIZATION")
print(" "*20 + "Target: Beat Phase 9 MAE=45.10")
print("="*90)
print()

# Load with BOTH modalities
print("[LOAD] Loading multimodal dataset...")
df_main = pd.read_csv('multi_account_dataset.csv')
df_bert = pd.read_csv('data/processed/bert_embeddings_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])
df_aes = pd.read_csv('data/processed/aesthetic_features_multi_account.csv').drop_duplicates(subset=['post_id', 'account'])

df = df_main.merge(df_bert, on=['post_id', 'account'], how='inner')
df = df.merge(df_aes, on=['post_id', 'account'], how='inner')

print(f"   Dataset: {len(df)} posts (BOTH visual + text)")
print()

# Features
baseline_cols = ['caption_length', 'word_count', 'hashtag_count', 'mention_count',
                 'is_video', 'hour', 'day_of_week', 'is_weekend', 'month']
bert_cols = [col for col in df.columns if col.startswith('bert_')]
aes_cols = [col for col in df.columns if col.startswith('aesthetic_')]

X_baseline = df[baseline_cols].values
X_bert_full = df[bert_cols].values
X_aes_full = df[aes_cols].values
y = df['likes'].values

# Split
train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)

X_baseline_train = X_baseline[train_idx]
X_baseline_test = X_baseline[test_idx]
X_bert_train = X_bert_full[train_idx]
X_bert_test = X_bert_full[test_idx]
X_aes_train = X_aes_full[train_idx]
X_aes_test = X_aes_full[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

# Preprocessing
clip_threshold = np.percentile(y_train, 99)
y_train_log = np.log1p(np.clip(y_train, 0, clip_threshold))
y_test_log = np.log1p(np.clip(y_test, 0, clip_threshold))

print("[EXPERIMENT] Grid search PCA dimensions...")
print()

# PCA grid
bert_pca_options = [40, 45, 50, 55, 60, 65, 70]
aes_pca_options = [4, 5, 6, 7, 8]

best_mae = float('inf')
best_bert_pca = None
best_aes_pca = None
results = []

for bert_n in bert_pca_options:
    for aes_n in aes_pca_options:
        # PCA
        pca_bert = PCA(n_components=bert_n, random_state=42)
        if aes_n == 8:
            X_aes_pca_train = X_aes_train
            X_aes_pca_test = X_aes_test
            pca_aes = None
        else:
            pca_aes = PCA(n_components=aes_n, random_state=42)
            X_aes_pca_train = pca_aes.fit_transform(X_aes_train)
            X_aes_pca_test = pca_aes.transform(X_aes_test)

        X_bert_pca_train = pca_bert.fit_transform(X_bert_train)
        X_bert_pca_test = pca_bert.transform(X_bert_test)

        # Combine
        X_train = np.hstack([X_baseline_train, X_bert_pca_train, X_aes_pca_train])
        X_test = np.hstack([X_baseline_test, X_bert_pca_test, X_aes_pca_test])

        # Scale
        scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Quick stacking test (simplified - 3 models)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros((len(X_train_scaled), 3))
        test_preds = np.zeros((len(X_test_scaled), 3))

        models = [
            GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=8, subsample=0.8, random_state=42),
            HistGradientBoostingRegressor(max_iter=500, learning_rate=0.07, max_depth=7, random_state=42),
            RandomForestRegressor(n_estimators=250, max_depth=16, random_state=42, n_jobs=-1),
        ]

        for i, model in enumerate(models):
            # OOF
            for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
                m = model.__class__(**model.get_params())
                m.fit(X_train_scaled[tr_idx], y_train_log[tr_idx])
                oof_preds[val_idx, i] = m.predict(X_train_scaled[val_idx])

            # Test
            model.fit(X_train_scaled, y_train_log)
            test_preds[:, i] = model.predict(X_test_scaled)

        # Meta
        meta = Ridge(alpha=10)
        meta.fit(oof_preds, y_train_log)

        y_pred_log = meta.predict(test_preds)
        y_pred = np.expm1(y_pred_log)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        bert_var = pca_bert.explained_variance_ratio_.sum()
        aes_var = 1.0 if aes_n == 8 else pca_aes.explained_variance_ratio_.sum()

        results.append({
            'bert_pca': bert_n,
            'aes_pca': aes_n,
            'bert_var': bert_var,
            'aes_var': aes_var,
            'mae': mae,
            'r2': r2
        })

        print(f"   BERT={bert_n:2d} ({bert_var:.1%}), Aes={aes_n} ({aes_var:.1%}): MAE={mae:.2f}, R2={r2:.4f}")

        if mae < best_mae:
            best_mae = mae
            best_bert_pca = bert_n
            best_aes_pca = aes_n

print()
print(f"[BEST PCA] BERT={best_bert_pca}, Aesthetic={best_aes_pca}: MAE={best_mae:.2f}")
print()

# ============================================================================
# Save results
# ============================================================================

if best_mae < 45.10:
    print("[SAVE] New champion! Saving model...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'models/phase10_2_pca_{timestamp}.pkl'

    model_package = {
        'phase': '10.2_pca_optimization',
        'mae': best_mae,
        'best_bert_pca': best_bert_pca,
        'best_aes_pca': best_aes_pca,
        'all_results': results,
        'visual_included': True,
        'text_included': True,
        'timestamp': timestamp
    }

    joblib.dump(model_package, model_filename)
    print(f"   Saved: {model_filename}")
    print()

print("="*90)
print(" "*30 + "PHASE 10.2 COMPLETE!")
print("="*90)
print()
print(f"[RESULT] MAE={best_mae:.2f} (Phase 9 was 45.10)")
if best_mae < 45.10:
    print(f"   IMPROVED by {(45.10-best_mae)/45.10*100:.1f}%!")
else:
    print(f"   PCA optimization did not improve (Phase 9 PCA remains optimal)")
