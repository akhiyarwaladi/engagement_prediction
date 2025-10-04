#!/usr/bin/env python3
"""
Prediction script for new Instagram posts.

Usage:
    python predict.py --caption "Your caption here" --hashtags 5 --video
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import BaselineModel
from src.utils import get_model_path


def extract_features(caption, hashtags, mentions, is_video, post_datetime):
    """Extract features from post data."""
    features = {}

    # Text features
    features['caption_length'] = len(caption)
    features['word_count'] = len(caption.split())

    # Social features
    features['hashtag_count'] = hashtags
    features['mention_count'] = mentions

    # Media features
    features['is_video'] = 1 if is_video else 0

    # Temporal features
    features['hour'] = post_datetime.hour
    features['day_of_week'] = post_datetime.weekday()
    features['is_weekend'] = 1 if post_datetime.weekday() >= 5 else 0
    features['month'] = post_datetime.month

    return features


def predict_engagement(caption, hashtags=5, mentions=0, is_video=False,
                       post_datetime=None):
    """Predict engagement for a post.

    Args:
        caption: Post caption text
        hashtags: Number of hashtags
        mentions: Number of mentions
        is_video: Whether content is video
        post_datetime: Posting datetime (default: now)

    Returns:
        Predicted number of likes
    """
    if post_datetime is None:
        post_datetime = datetime.now()

    # Load model
    model_path = get_model_path('baseline_rf_model.pkl')

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Please train the model first: python run_pipeline.py"
        )

    model = BaselineModel.load(model_path)

    # Extract features
    features = extract_features(caption, hashtags, mentions, is_video, post_datetime)

    # Convert to DataFrame
    X = pd.DataFrame([features])

    # Predict
    predicted_likes = model.predict(X)[0]

    return predicted_likes, features


def get_recommendations(features, predicted_likes):
    """Generate recommendations."""
    recommendations = []

    # Time recommendations
    if features['hour'] < 8 or features['hour'] > 22:
        recommendations.append(
            "⏰ Consider posting between 8 AM - 10 PM for better engagement"
        )

    if features['is_weekend']:
        recommendations.append(
            "📅 Weekend posts typically get less engagement"
        )

    # Content recommendations
    if features['caption_length'] < 50:
        recommendations.append(
            "📝 Caption is short. Longer captions (100-200 chars) often perform better"
        )

    if features['hashtag_count'] == 0:
        recommendations.append(
            "#️⃣ Add hashtags! 3-5 relevant hashtags can improve engagement"
        )

    if features['hashtag_count'] > 10:
        recommendations.append(
            "❌ Too many hashtags. Try 5-7 most relevant ones"
        )

    if not features['is_video']:
        recommendations.append(
            "🎥 Videos often get 20-30% more engagement than photos"
        )

    # Optimal times
    if features['hour'] in [10, 11, 12, 17, 18, 19]:
        recommendations.append(
            "✅ Good timing! This is typically a high-engagement hour"
        )

    return recommendations


def main():
    """Main prediction interface."""
    parser = argparse.ArgumentParser(
        description='Predict Instagram post engagement'
    )

    parser.add_argument(
        '--caption',
        type=str,
        required=True,
        help='Post caption text'
    )

    parser.add_argument(
        '--hashtags',
        type=int,
        default=5,
        help='Number of hashtags (default: 5)'
    )

    parser.add_argument(
        '--mentions',
        type=int,
        default=0,
        help='Number of mentions (default: 0)'
    )

    parser.add_argument(
        '--video',
        action='store_true',
        help='Content is video (default: False)'
    )

    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Posting datetime in format: YYYY-MM-DD HH:MM (default: now)'
    )

    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed feature values'
    )

    args = parser.parse_args()

    # Parse datetime
    if args.date:
        try:
            post_datetime = datetime.strptime(args.date, '%Y-%m-%d %H:%M')
        except ValueError:
            print("❌ Invalid date format. Use: YYYY-MM-DD HH:MM")
            return 1
    else:
        post_datetime = datetime.now()

    print("\n" + "=" * 80)
    print(" " * 25 + "INSTAGRAM ENGAGEMENT PREDICTION")
    print("=" * 80)

    print("\n📝 Post Details:")
    print("-" * 80)
    print(f"Caption: {args.caption[:60]}..." if len(args.caption) > 60 else f"Caption: {args.caption}")
    print(f"Hashtags: {args.hashtags}")
    print(f"Mentions: {args.mentions}")
    print(f"Content Type: {'🎥 Video' if args.video else '📷 Photo'}")
    print(f"Posting Time: {post_datetime.strftime('%Y-%m-%d %H:%M')} ({post_datetime.strftime('%A')})")
    print("-" * 80)

    try:
        # Make prediction
        predicted_likes, features = predict_engagement(
            args.caption,
            args.hashtags,
            args.mentions,
            args.video,
            post_datetime
        )

        # Calculate engagement rate
        FOLLOWER_COUNT = 4631
        engagement_rate = (predicted_likes / FOLLOWER_COUNT) * 100

        # Categorize performance
        if predicted_likes < 150:
            performance = "Low"
            emoji = "🔴"
        elif predicted_likes < 300:
            performance = "Medium"
            emoji = "🟡"
        else:
            performance = "High"
            emoji = "🟢"

        # Display prediction
        print("\n📊 Prediction Results:")
        print("-" * 80)
        print(f"  Predicted Likes: {predicted_likes:.0f}")
        print(f"  Engagement Rate: {engagement_rate:.2f}%")
        print(f"  Performance: {emoji} {performance}")
        print("-" * 80)

        # Show detailed features if requested
        if args.detailed:
            print("\n🔍 Feature Values:")
            print("-" * 80)
            for feat, val in features.items():
                print(f"  {feat:20s}: {val}")
            print("-" * 80)

        # Get recommendations
        recommendations = get_recommendations(features, predicted_likes)

        if recommendations:
            print("\n💡 Recommendations:")
            print("-" * 80)
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
            print("-" * 80)

        print("\n" + "=" * 80)
        print("DONE! 🎉")
        print("=" * 80 + "\n")

        return 0

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        return 1

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
