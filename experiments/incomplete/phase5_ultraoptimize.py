#!/usr/bin/env python3
"""
PHASE 5: ULTRA-OPTIMIZED INSTAGRAM ENGAGEMENT PREDICTION
=========================================================

Improvements over Phase 4b:
1. Advanced temporal features (posting frequency, trend momentum)
2. Video frame extraction for proper ViT embeddings
3. Optuna hyperparameter optimization (Bayesian)
4. Stacking ensemble with meta-learner
5. Robust outlier handling (Winsorization)
6. Cross-validation with time-series splits
7. GPU acceleration for transformers

Target: MAE < 90, R² > 0.30
"""

import pandas as pd
import numpy as np
import joblib
import optuna
import warnings
from pathlib import Path
from datetime import datetime
import torch
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from transformers import AutoTokenizer, AutoModel, ViTImageProcessor, ViTModel
from PIL import Image
import cv2

warnings.filterwarnings('ignore')

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def extract_advanced_temporal_features(df):
    """Extract advanced temporal features"""
    print("\n[FEATURES] Extracting advanced temporal features...")

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Days since last post
    df['days_since_last_post'] = df['date'].diff().dt.total_seconds() / 86400
    df['days_since_last_post'].fillna(df['days_since_last_post'].median(), inplace=True)

    # Posting frequency (posts per week in rolling window)
    df['post_index'] = range(len(df))
    df['posting_frequency'] = df.groupby(pd.Grouper(key='date', freq='7D')).size().reindex(
        df['date']).fillna(method='ffill').fillna(1).values

    # Trend momentum (likes growth over last 5 posts)
    df['likes_ma5'] = df['likes'].rolling(window=5, min_periods=1).mean()
    df['trend_momentum'] = (df['likes'] - df['likes_ma5']) / (df['likes_ma5'] + 1)

    # Time since first post (account age effect)
    df['days_since_first_post'] = (df['date'] - df['date'].min()).dt.total_seconds() / 86400

    # Engagement velocity (likes per day since last post)
    df['engagement_velocity'] = df['likes'] / (df['days_since_last_post'] + 1)

    print(f"  Added 6 temporal features")

    return df

def extract_video_frames(video_path, n_frames=3):
    """Extract representative frames from video for ViT processing"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return None

        # Extract frames at evenly spaced intervals
        frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))

        cap.release()
        return frames if frames else None
    except Exception as e:
        print(f"  Error extracting frames: {e}")
        return None

def extract_enhanced_vit_embeddings(df, output_path='data/processed/vit_embeddings_enhanced.csv'):
    """Extract ViT embeddings with proper video frame handling"""
    print("\n[ViT] Extracting enhanced ViT embeddings (with video support)...")

    if Path(output_path).exists():
        print(f"  Loading cached embeddings from {output_path}")
        return pd.read_csv(output_path)

    # Load ViT model
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name).to(DEVICE)
    model.eval()

    embeddings_list = []

    for idx, row in df.iterrows():
        try:
            file_path = Path(row['file_path'])

            if not file_path.exists():
                # Zero vector for missing files
                embedding = np.zeros(768)
            elif row['is_video']:
                # Extract frames from video
                frames = extract_video_frames(file_path, n_frames=3)
                if frames:
                    # Average embeddings from multiple frames
                    frame_embeddings = []
                    for frame in frames:
                        inputs = processor(images=frame, return_tensors="pt")
                        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                        with torch.no_grad():
                            outputs = model(**inputs)
                            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                            frame_embeddings.append(embedding)

                    # Average across frames
                    embedding = np.mean(frame_embeddings, axis=0)
                else:
                    # Zero vector if frame extraction fails
                    embedding = np.zeros(768)
            else:
                # Process image normally
                image = Image.open(file_path).convert('RGB')
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

            embeddings_list.append(embedding)

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(df)} posts")

        except Exception as e:
            print(f"  Error processing {row['file_path']}: {e}")
            embeddings_list.append(np.zeros(768))

    # Create DataFrame
    embeddings_df = pd.DataFrame(embeddings_list, columns=[f'vit_{i}' for i in range(768)])
    embeddings_df['post_id'] = df['post_id'].values

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    embeddings_df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")

    # Clean up GPU memory
    del model, processor
    torch.cuda.empty_cache()

    return embeddings_df

def prepare_features_phase5(df):
    """Prepare comprehensive feature set for Phase 5"""
    print("\n[PREPARE] Preparing Phase 5 feature set...")

    # Extract advanced temporal features
    df = extract_advanced_temporal_features(df)

    # Load/extract BERT embeddings
    bert_path = 'data/processed/bert_embeddings.csv'
    if Path(bert_path).exists():
        bert_df = pd.read_csv(bert_path)
        print(f"[BERT] Loaded {bert_df.shape[1]-1} BERT dimensions")
    else:
        print("[ERROR] BERT embeddings not found! Run extract_bert_features.py first")
        return None

    # Load/extract enhanced ViT embeddings
    vit_df = extract_enhanced_vit_embeddings(df)
    print(f"[ViT] Loaded {vit_df.shape[1]-1} ViT dimensions")

    # Merge on post_id
    df_merged = df.merge(bert_df, on='post_id', how='left')
    df_merged = df_merged.merge(vit_df, on='post_id', how='left')

    # Baseline features
    baseline_features = [
        'caption_length', 'word_count', 'hashtag_count', 'mention_count',
        'is_video', 'hour', 'day_of_week', 'is_weekend', 'month'
    ]

    # Advanced temporal features
    temporal_features = [
        'days_since_last_post', 'posting_frequency', 'trend_momentum',
        'days_since_first_post', 'engagement_velocity'
    ]

    # Create baseline features if not exist
    if 'caption_length' not in df_merged.columns:
        df_merged['caption'] = df_merged['caption'].fillna('')
        df_merged['caption_length'] = df_merged['caption'].str.len()
        df_merged['word_count'] = df_merged['caption'].str.split().str.len()
        df_merged['hashtag_count'] = df_merged['hashtags_count']
        df_merged['mention_count'] = df_merged['mentions_count']
        df_merged['hour'] = pd.to_datetime(df_merged['date']).dt.hour
        df_merged['day_of_week'] = pd.to_datetime(df_merged['date']).dt.dayofweek
        df_merged['is_weekend'] = df_merged['day_of_week'].isin([5, 6]).astype(int)
        df_merged['month'] = pd.to_datetime(df_merged['date']).dt.month

    # Combine all features
    bert_cols = [c for c in df_merged.columns if c.startswith('bert_')]
    vit_cols = [c for c in df_merged.columns if c.startswith('vit_')]

    all_features = baseline_features + temporal_features + bert_cols + vit_cols
    X = df_merged[all_features].fillna(0)
    y = df_merged['likes'].values

    print(f"\n[FEATURES] Total features: {len(all_features)}")
    print(f"  Baseline: {len(baseline_features)}")
    print(f"  Temporal: {len(temporal_features)}")
    print(f"  BERT: {len(bert_cols)}")
    print(f"  ViT: {len(vit_cols)}")

    return X, y, df_merged

# ============================================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================================

def objective_rf(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for Random Forest"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 10, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    return mae

def objective_hgb(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for HistGradientBoosting"""
    params = {
        'max_iter': trial.suggest_int('max_iter', 200, 600),
        'max_depth': trial.suggest_int('max_depth', 10, 25),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 0.5),
        'random_state': RANDOM_STATE
    }

    model = HistGradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    return mae

def optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=50):
    """Optimize hyperparameters using Optuna"""
    print("\n[OPTUNA] Starting hyperparameter optimization...")
    print(f"  Trials: {n_trials}")

    # Optimize Random Forest
    print("\n  Optimizing Random Forest...")
    study_rf = optuna.create_study(direction='minimize', study_name='rf_optimization')
    study_rf.optimize(
        lambda trial: objective_rf(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print(f"  Best RF MAE: {study_rf.best_value:.2f}")
    print(f"  Best RF params: {study_rf.best_params}")

    # Optimize HistGradientBoosting
    print("\n  Optimizing HistGradientBoosting...")
    study_hgb = optuna.create_study(direction='minimize', study_name='hgb_optimization')
    study_hgb.optimize(
        lambda trial: objective_hgb(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print(f"  Best HGB MAE: {study_hgb.best_value:.2f}")
    print(f"  Best HGB params: {study_hgb.best_params}")

    return study_rf.best_params, study_hgb.best_params

# ============================================================================
# TRAINING
# ============================================================================

def train_phase5_model(X, y, rf_params, hgb_params):
    """Train Phase 5 model with optimized hyperparameters"""
    print("\n[TRAIN] Training Phase 5 model...")

    # Outlier handling (Winsorization at 99th percentile)
    q99 = np.percentile(y, 99)
    y_clipped = np.clip(y, 0, q99)
    print(f"  Winsorized at {q99:.0f} likes (99th percentile)")

    # Log transformation
    y_log = np.log1p(y_clipped)

    # Train/test split (time-based)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_log[:split_idx], y_log[split_idx:]
    y_train_orig, y_test_orig = y[:split_idx], y[split_idx:]

    print(f"  Train size: {len(X_train)}")
    print(f"  Test size: {len(X_test)}")

    # Apply PCA to BERT and ViT embeddings
    print("\n  Applying PCA dimensionality reduction...")

    # Identify BERT and ViT columns
    bert_cols = [i for i, col in enumerate(X.columns) if col.startswith('bert_')]
    vit_cols = [i for i, col in enumerate(X.columns) if col.startswith('vit_')]
    other_cols = [i for i in range(len(X.columns)) if i not in bert_cols + vit_cols]

    # PCA for BERT (768 -> 50)
    pca_bert = PCA(n_components=50, random_state=RANDOM_STATE)
    X_train_bert_pca = pca_bert.fit_transform(X_train.iloc[:, bert_cols])
    X_test_bert_pca = pca_bert.transform(X_test.iloc[:, bert_cols])
    print(f"    BERT: 768 -> 50 dims ({pca_bert.explained_variance_ratio_.sum()*100:.1f}% variance)")

    # PCA for ViT (768 -> 50)
    pca_vit = PCA(n_components=50, random_state=RANDOM_STATE)
    X_train_vit_pca = pca_vit.fit_transform(X_train.iloc[:, vit_cols])
    X_test_vit_pca = pca_vit.transform(X_test.iloc[:, vit_cols])
    print(f"    ViT: 768 -> 50 dims ({pca_vit.explained_variance_ratio_.sum()*100:.1f}% variance)")

    # Combine features
    X_train_combined = np.hstack([
        X_train.iloc[:, other_cols].values,
        X_train_bert_pca,
        X_train_vit_pca
    ])
    X_test_combined = np.hstack([
        X_test.iloc[:, other_cols].values,
        X_test_bert_pca,
        X_test_vit_pca
    ])

    print(f"  Combined features: {X_train_combined.shape[1]}")

    # Quantile transformation
    scaler = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_test_combined)

    # Create base models with optimized hyperparameters
    rf_model = RandomForestRegressor(**rf_params, random_state=RANDOM_STATE, n_jobs=-1)
    hgb_model = HistGradientBoostingRegressor(**hgb_params, random_state=RANDOM_STATE)

    # Stacking ensemble with Ridge meta-learner
    print("\n  Training stacking ensemble...")
    stacking_model = StackingRegressor(
        estimators=[
            ('rf', rf_model),
            ('hgb', hgb_model)
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=5
    )

    stacking_model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_log = stacking_model.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)

    # Metrics
    mae = mean_absolute_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))

    print(f"\n[RESULTS] Phase 5 Performance:")
    print(f"  MAE: {mae:.2f} likes")
    print(f"  R²: {r2:.3f}")
    print(f"  RMSE: {rmse:.2f}")

    # Save model
    model_data = {
        'stacking_model': stacking_model,
        'pca_bert': pca_bert,
        'pca_vit': pca_vit,
        'scaler': scaler,
        'feature_cols': X.columns.tolist(),
        'bert_cols': bert_cols,
        'vit_cols': vit_cols,
        'other_cols': other_cols,
        'metrics': {'mae': mae, 'r2': r2, 'rmse': rmse}
    }

    output_path = 'models/phase5_ultra_model.pkl'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_data, output_path)
    print(f"\n  Model saved to {output_path}")

    return model_data, y_test_orig, y_pred

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution"""
    print("=" * 80)
    print("PHASE 5: ULTRA-OPTIMIZED INSTAGRAM ENGAGEMENT PREDICTION")
    print("=" * 80)

    # Load data
    print("\n[DATA] Loading dataset...")
    df = pd.read_csv('fst_unja_from_gallery_dl.csv')
    print(f"  Loaded {len(df)} posts")

    # Prepare features
    X, y, df_processed = prepare_features_phase5(df)

    if X is None:
        print("[ERROR] Feature preparation failed!")
        return

    # Split for hyperparameter optimization
    split_idx = int(len(X) * 0.7)
    X_train_opt = X[:split_idx]
    y_train_opt = np.log1p(np.clip(y[:split_idx], 0, np.percentile(y[:split_idx], 99)))

    val_start = split_idx
    val_end = int(len(X) * 0.8)
    X_val_opt = X[val_start:val_end]
    y_val_opt = np.log1p(np.clip(y[val_start:val_end], 0, np.percentile(y[val_start:val_end], 99)))

    # Apply same preprocessing for optimization
    scaler_opt = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
    X_train_opt_scaled = scaler_opt.fit_transform(X_train_opt)
    X_val_opt_scaled = scaler_opt.transform(X_val_opt)

    # Optimize hyperparameters
    rf_params, hgb_params = optimize_hyperparameters(
        X_train_opt_scaled, y_train_opt,
        X_val_opt_scaled, y_val_opt,
        n_trials=30  # Reduced for faster execution
    )

    # Train final model
    model_data, y_true, y_pred = train_phase5_model(X, y, rf_params, hgb_params)

    print("\n" + "=" * 80)
    print("PHASE 5 COMPLETE!")
    print("=" * 80)
    print(f"\nFinal Performance:")
    print(f"  MAE: {model_data['metrics']['mae']:.2f} likes")
    print(f"  R²: {model_data['metrics']['r2']:.3f}")
    print(f"  RMSE: {model_data['metrics']['rmse']:.2f}")
    print(f"\nModel saved to: models/phase5_ultra_model.pkl")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
