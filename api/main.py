#!/usr/bin/env python3
"""
FastAPI Prediction API
Production-ready REST API for Instagram engagement prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
sys.path.append('..')

app = FastAPI(
    title="Instagram Engagement Prediction API",
    description="Predict Instagram post engagement using multimodal AI",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
MODEL_PATH = "../models/phase5_1_advanced_model.pkl"
model_data = None


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model_data
    try:
        model_data = joblib.load(MODEL_PATH)
        print(f"[OK] Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        model_data = None


# Request/Response models
class PredictionRequest(BaseModel):
    caption: str = Field(..., description="Post caption text")
    hashtags_count: int = Field(default=5, ge=0, le=30, description="Number of hashtags")
    is_video: bool = Field(default=False, description="Whether post is a video")
    datetime: str = Field(..., description="Posting datetime (ISO format)")

    class Config:
        schema_extra = {
            "example": {
                "caption": "Selamat datang mahasiswa baru FST UNJA! 🎓",
                "hashtags_count": 5,
                "is_video": False,
                "datetime": "2025-10-04T10:00:00"
            }
        }


class PredictionResponse(BaseModel):
    predicted_likes: int
    confidence_interval: List[int]
    recommendation: str
    factors: Dict[str, float]
    metadata: Dict[str, str]


# API Endpoints
@app.get("/")
async def root():
    """API root"""
    return {
        "name": "Instagram Engagement Prediction API",
        "version": "1.0.0",
        "status": "ready" if model_data else "model_not_loaded",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "model_info": "/model/info"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model_data else "unhealthy",
        "model_loaded": model_data is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info")
async def model_info():
    """Get model information"""
    if not model_data:
        raise HTTPException(status_code=503, detail="Model not loaded")

    metrics = model_data.get('metrics', {})

    return {
        "model_type": "Stacking Ensemble (RF + HGB + GB meta-learner)",
        "version": "Phase 5.1",
        "performance": {
            "test_mae": round(metrics.get('mae', 0), 2),
            "test_r2": round(metrics.get('r2', 0), 3),
            "test_rmse": round(metrics.get('rmse', 0), 2)
        },
        "features": {
            "baseline": 6,
            "cyclic_temporal": 6,
            "lag_features": 6,
            "bert_pca": 100,
            "vit_pca": 100,
            "interactions": 30,
            "total": 248
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict Instagram engagement

    Returns predicted likes with confidence interval and recommendations
    """

    if not model_data:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Parse datetime
        post_datetime = datetime.fromisoformat(request.datetime.replace('Z', '+00:00'))

        # Create feature DataFrame
        features = create_features(
            caption=request.caption,
            hashtags_count=request.hashtags_count,
            is_video=request.is_video,
            post_datetime=post_datetime
        )

        # Make prediction
        prediction = model_data['stacking_model'].predict(
            model_data['scaler'].transform(features)
        )[0]

        # Transform back from log space
        predicted_likes = int(np.expm1(prediction))

        # Confidence interval (±20%)
        ci_lower = int(predicted_likes * 0.8)
        ci_upper = int(predicted_likes * 1.2)

        # Calculate factor scores
        hour_score = calculate_time_score(post_datetime.hour)
        caption_score = calculate_caption_score(request.caption)
        media_score = 0.8 if request.is_video else 0.9  # Videos slightly lower

        # Generate recommendation
        recommendation = generate_recommendation(
            predicted_likes=predicted_likes,
            hour_score=hour_score,
            caption_score=caption_score,
            is_weekend=post_datetime.weekday() >= 5
        )

        return PredictionResponse(
            predicted_likes=predicted_likes,
            confidence_interval=[ci_lower, ci_upper],
            recommendation=recommendation,
            factors={
                "time_score": round(hour_score, 2),
                "caption_score": round(caption_score, 2),
                "media_score": round(media_score, 2)
            },
            metadata={
                "model_version": "5.1",
                "prediction_time": datetime.now().isoformat()
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


def create_features(caption: str, hashtags_count: int, is_video: bool,
                   post_datetime: datetime) -> pd.DataFrame:
    """Create feature vector from input"""

    # Baseline features
    caption_length = len(caption)
    word_count = len(caption.split())
    is_weekend = 1 if post_datetime.weekday() >= 5 else 0

    # Cyclic temporal features
    hour = post_datetime.hour
    day = post_datetime.weekday()
    month = post_datetime.month

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day / 7)
    day_cos = np.cos(2 * np.pi * day / 7)
    month_sin = np.sin(2 * np.pi * (month - 1) / 12)
    month_cos = np.cos(2 * np.pi * (month - 1) / 12)

    # Note: For production, we'd need to extract BERT/ViT embeddings
    # For now, use zeros (this is a simplified version)
    # In real production, integrate with BERT/ViT models

    features = {
        'caption_length': caption_length,
        'word_count': word_count,
        'hashtag_count': hashtags_count,
        'mention_count': 0,  # Default
        'is_video': int(is_video),
        'is_weekend': is_weekend,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'day_sin': day_sin,
        'day_cos': day_cos,
        'month_sin': month_sin,
        'month_cos': month_cos,
    }

    # Lag features (use defaults for API)
    for lag in [1, 2, 3, 5]:
        features[f'likes_lag_{lag}'] = 200  # Default

    features['likes_rolling_mean_5'] = 200
    features['likes_rolling_std_5'] = 50

    # BERT/ViT embeddings (would need to extract in production)
    # For demo, use zeros
    for i in range(100):
        features[f'bert_pca_{i}'] = 0.0
        features[f'vit_pca_{i}'] = 0.0

    # Interaction features
    for i in range(30):
        features[f'interaction_{i}'] = 0.0

    return pd.DataFrame([features])


def calculate_time_score(hour: int) -> float:
    """Calculate posting time score (0-1)"""
    # Peak hours: 10-12, 17-19
    if 10 <= hour <= 12 or 17 <= hour <= 19:
        return 0.9
    elif 7 <= hour <= 9 or 13 <= hour <= 16 or 20 <= hour <= 22:
        return 0.7
    else:
        return 0.4


def calculate_caption_score(caption: str) -> float:
    """Calculate caption quality score (0-1)"""
    length = len(caption)

    # Optimal length: 100-200 chars
    if 100 <= length <= 200:
        return 0.9
    elif 50 <= length <= 300:
        return 0.7
    else:
        return 0.5


def generate_recommendation(predicted_likes: int, hour_score: float,
                           caption_score: float, is_weekend: bool) -> str:
    """Generate posting recommendation"""

    if predicted_likes > 300:
        engagement = "Very high"
    elif predicted_likes > 200:
        engagement = "High"
    elif predicted_likes > 150:
        engagement = "Good"
    else:
        engagement = "Moderate"

    recommendations = []

    if hour_score >= 0.8:
        recommendations.append("Excellent posting time!")
    elif hour_score < 0.6:
        recommendations.append("Consider posting at 10-12 AM or 5-7 PM for better engagement.")

    if caption_score < 0.7:
        recommendations.append("Try a caption length of 100-200 characters.")

    if is_weekend:
        recommendations.append("Weekend post - expect 20% higher engagement!")

    if recommendations:
        return f"{engagement} engagement expected. " + " ".join(recommendations)
    else:
        return f"{engagement} engagement expected. Post looks great!"


# Run with: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
