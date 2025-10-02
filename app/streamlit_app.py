"""Streamlit web application for Instagram engagement prediction.

Simple MVP interface for FST UNJA social media team.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import BaselineModel
from src.utils import get_model_path, get_project_root, load_config


# Page config
st.set_page_config(
    page_title="Instagram Engagement Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model():
    """Load trained model."""
    try:
        model_path = get_model_path('baseline_rf_model.pkl')
        model = BaselineModel.load(model_path)
        return model, None
    except Exception as e:
        return None, str(e)


@st.cache_data
def load_historical_data():
    """Load historical post data."""
    try:
        data_path = get_project_root() / 'data' / 'processed' / 'baseline_dataset.csv'
        df = pd.read_csv(data_path)
        return df, None
    except Exception as e:
        return None, str(e)


def extract_features_from_input(caption, hashtag_count, mention_count, is_video,
                                 post_date, post_time):
    """Extract features from user input."""
    features = {}

    # Text features
    features['caption_length'] = len(caption)
    features['word_count'] = len(caption.split())

    # Hashtag/mention features
    features['hashtag_count'] = hashtag_count
    features['mention_count'] = mention_count

    # Media features
    features['is_video'] = 1 if is_video else 0

    # Temporal features
    import datetime
    dt = datetime.datetime.combine(post_date, post_time)
    features['hour'] = dt.hour
    features['day_of_week'] = dt.weekday()
    features['is_weekend'] = 1 if dt.weekday() >= 5 else 0
    features['month'] = dt.month

    return features


def get_recommendations(features, predicted_likes):
    """Generate simple recommendations based on features."""
    recommendations = []

    # Time-based recommendations
    if features['hour'] < 8 or features['hour'] > 22:
        recommendations.append("⏰ Consider posting between 8 AM - 10 PM for better engagement")

    if features['day_of_week'] >= 5:  # Weekend
        recommendations.append("📅 Weekend posts typically get less engagement. Consider posting on weekdays.")

    # Content recommendations
    if features['caption_length'] < 50:
        recommendations.append("📝 Caption is quite short. Longer captions (100-200 chars) often perform better.")

    if features['hashtag_count'] == 0:
        recommendations.append("#️⃣ Add hashtags! Posts with 3-5 relevant hashtags typically get more engagement.")

    if features['hashtag_count'] > 10:
        recommendations.append("❌ Too many hashtags. Try reducing to 5-7 most relevant ones.")

    # Best practices
    if not features['is_video']:
        recommendations.append("🎥 Videos typically get 20-30% more engagement than photos. Consider using video content.")

    # Optimal posting times
    if features['hour'] in [10, 11, 12, 17, 18, 19]:
        recommendations.append("✅ Good timing! This is typically a high-engagement hour.")

    if len(recommendations) == 0:
        recommendations.append("✅ Your post settings look good! No specific recommendations.")

    return recommendations


def main():
    """Main Streamlit app."""

    # Header
    st.title("📊 Instagram Engagement Predictor")
    st.markdown("### FST UNJA - Prediksi Engagement Postingan Instagram")
    st.markdown("---")

    # Load model
    model, error = load_model()

    if error:
        st.error(f"❌ Error loading model: {error}")
        st.info("💡 Please train the model first by running: `python -m src.models.trainer`")
        return

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Instagram Engagement Predictor** menggunakan Machine Learning untuk
        memprediksi jumlah likes yang akan didapat oleh postingan Instagram.

        **Model:** Random Forest
        **Features:** 9 fitur (text, temporal, media type)
        **Dataset:** 227 posts from @fst_unja

        ---
        **Research Team:**
        - Jefri Marzal (PI)
        - Muhammad Razi A.
        - Miranty Yudistira
        - Akhiyar Waladi
        - Hamzah Alghifari
        """)

        st.markdown("---")
        st.markdown("**Version:** 1.0 (MVP)")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Predict", "📈 Analytics", "🎓 Insights"])

    # TAB 1: PREDICTION
    with tab1:
        st.header("Prediksi Engagement Postingan")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Caption input
            caption = st.text_area(
                "Caption",
                placeholder="Tuliskan caption postingan Instagram...",
                height=150,
                help="Caption yang akan digunakan untuk postingan"
            )

            # Hashtag and mentions
            col_a, col_b = st.columns(2)
            with col_a:
                hashtag_count = st.number_input(
                    "Jumlah Hashtag",
                    min_value=0,
                    max_value=30,
                    value=5,
                    help="Berapa banyak hashtag yang akan digunakan"
                )

            with col_b:
                mention_count = st.number_input(
                    "Jumlah Mention (@)",
                    min_value=0,
                    max_value=10,
                    value=0,
                    help="Berapa banyak mention/tag yang akan digunakan"
                )

        with col2:
            # Media type
            is_video = st.checkbox("📹 Video Content?", help="Centang jika konten berupa video")

            # Date and time
            post_date = st.date_input("📅 Tanggal Posting", help="Pilih tanggal posting")
            post_time = st.time_input("⏰ Jam Posting", help="Pilih jam posting")

            st.markdown("---")

            # Predict button
            predict_button = st.button("🚀 Prediksi Engagement", type="primary", use_container_width=True)

        if predict_button:
            if not caption:
                st.warning("⚠️ Mohon masukkan caption terlebih dahulu!")
            else:
                with st.spinner("Menghitung prediksi..."):
                    # Extract features
                    features = extract_features_from_input(
                        caption, hashtag_count, mention_count, is_video,
                        post_date, post_time
                    )

                    # Convert to DataFrame
                    X_input = pd.DataFrame([features])

                    # Predict
                    predicted_likes = model.predict(X_input)[0]

                    # Display result
                    st.markdown("---")
                    st.markdown("### 📊 Hasil Prediksi")

                    # Metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            label="Prediksi Likes",
                            value=f"{predicted_likes:.0f}",
                            help="Estimasi jumlah likes yang akan didapat"
                        )

                    with col2:
                        engagement_rate = (predicted_likes / 4631) * 100  # Follower count
                        st.metric(
                            label="Engagement Rate",
                            value=f"{engagement_rate:.2f}%",
                            help="Persentase engagement dibanding follower"
                        )

                    with col3:
                        # Categorize performance
                        if predicted_likes < 150:
                            performance = "Low"
                            color = "🔴"
                        elif predicted_likes < 300:
                            performance = "Medium"
                            color = "🟡"
                        else:
                            performance = "High"
                            color = "🟢"

                        st.metric(
                            label="Performance",
                            value=f"{color} {performance}",
                            help="Kategori performa prediksi"
                        )

                    # Recommendations
                    st.markdown("### 💡 Rekomendasi")
                    recommendations = get_recommendations(features, predicted_likes)

                    for rec in recommendations:
                        st.info(rec)

    # TAB 2: ANALYTICS
    with tab2:
        st.header("Analisis Data Historis")

        df, error = load_historical_data()

        if error:
            st.error(f"❌ Error loading data: {error}")
        else:
            # Summary stats
            st.subheader("📊 Statistik Ringkasan")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Posts", len(df))

            with col2:
                st.metric("Avg Likes", f"{df['likes'].mean():.0f}")

            with col3:
                st.metric("Max Likes", f"{df['likes'].max():.0f}")

            with col4:
                st.metric("Avg Engagement", f"{df['engagement_rate'].mean():.2f}%")

            st.markdown("---")

            # Visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Engagement over time
                fig_time = px.line(
                    df.sort_values('date'),
                    x='date',
                    y='likes',
                    title='Engagement Trend Over Time',
                    labels={'likes': 'Likes', 'date': 'Date'}
                )
                st.plotly_chart(fig_time, use_container_width=True)

            with col2:
                # Photo vs Video
                fig_media = px.box(
                    df,
                    x='is_video',
                    y='likes',
                    title='Likes by Media Type',
                    labels={'is_video': 'Media Type', 'likes': 'Likes'}
                )
                fig_media.update_xaxes(ticktext=['Photo', 'Video'], tickvals=[0, 1])
                st.plotly_chart(fig_media, use_container_width=True)

            # Posting time analysis
            st.subheader("⏰ Analisis Waktu Posting")

            col1, col2 = st.columns(2)

            with col1:
                # By hour
                hour_stats = df.groupby('hour')['likes'].mean().reset_index()
                fig_hour = px.bar(
                    hour_stats,
                    x='hour',
                    y='likes',
                    title='Average Likes by Hour',
                    labels={'hour': 'Hour', 'likes': 'Avg Likes'}
                )
                st.plotly_chart(fig_hour, use_container_width=True)

            with col2:
                # By day of week
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_stats = df.groupby('day_of_week')['likes'].mean().reset_index()
                day_stats['day_name'] = day_stats['day_of_week'].map(lambda x: day_names[x])

                fig_day = px.bar(
                    day_stats,
                    x='day_name',
                    y='likes',
                    title='Average Likes by Day of Week',
                    labels={'day_name': 'Day', 'likes': 'Avg Likes'}
                )
                st.plotly_chart(fig_day, use_container_width=True)

            # Top posts
            st.subheader("🏆 Top 10 Posts")
            top_posts = df.nlargest(10, 'likes')[['date', 'likes', 'is_video', 'hour', 'day_of_week']].copy()
            top_posts['media_type'] = top_posts['is_video'].map({0: '📷 Photo', 1: '🎥 Video'})
            top_posts = top_posts.drop('is_video', axis=1)

            st.dataframe(
                top_posts,
                use_container_width=True,
                hide_index=True
            )

    # TAB 3: INSIGHTS
    with tab3:
        st.header("📚 Key Insights & Best Practices")

        df, error = load_historical_data()

        if not error:
            st.subheader("🎯 Optimal Posting Strategy")

            # Calculate insights
            avg_by_hour = df.groupby('hour')['likes'].mean().sort_values(ascending=False)
            best_hours = avg_by_hour.head(3).index.tolist()

            avg_by_day = df.groupby('day_of_week')['likes'].mean().sort_values(ascending=False)
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            best_days = [day_names[d] for d in avg_by_day.head(3).index.tolist()]

            video_avg = df[df['is_video'] == 1]['likes'].mean()
            photo_avg = df[df['is_video'] == 0]['likes'].mean()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### ⏰ Best Posting Times")
                for hour in best_hours:
                    st.success(f"✅ {hour}:00 - {hour+1}:00 (avg: {avg_by_hour[hour]:.0f} likes)")

                st.markdown("### 📅 Best Days to Post")
                for i, day in enumerate(best_days, 1):
                    avg_likes = avg_by_day.iloc[i-1]
                    st.success(f"✅ {day} (avg: {avg_likes:.0f} likes)")

            with col2:
                st.markdown("### 📹 Content Type Performance")
                st.info(f"🎥 Video avg: **{video_avg:.0f} likes**")
                st.info(f"📷 Photo avg: **{photo_avg:.0f} likes**")

                if video_avg > photo_avg:
                    improvement = ((video_avg - photo_avg) / photo_avg) * 100
                    st.success(f"✅ Videos perform {improvement:.1f}% better than photos")
                else:
                    improvement = ((photo_avg - video_avg) / video_avg) * 100
                    st.success(f"✅ Photos perform {improvement:.1f}% better than videos")

            st.markdown("---")

            st.subheader("📝 General Best Practices")

            st.markdown("""
            1. **Timing Matters:** Post during peak hours (typically 10-12 AM or 5-7 PM)
            2. **Consistency:** Regular posting schedule helps build audience engagement
            3. **Hashtags:** Use 5-7 relevant hashtags for better discoverability
            4. **Content Mix:** Balance between photos and videos for variety
            5. **Caption Length:** 100-200 characters tends to perform well
            6. **Academic Events:** Posts during graduation/registration periods get more engagement
            7. **Weekend Effect:** Weekday posts generally perform better than weekend posts
            """)

        # Model info
        st.markdown("---")
        st.subheader("🤖 About the Model")

        st.markdown("""
        **Model Type:** Random Forest Regressor
        **Features Used:** 9 baseline features
        - Text features: caption length, word count
        - Social features: hashtag count, mention count
        - Media features: content type (photo/video)
        - Temporal features: hour, day of week, weekend flag, month

        **Training Data:** 227 Instagram posts from @fst_unja (2022-2024)
        **Performance:** MAE < 70 likes, R² > 0.50

        **Limitations:**
        - Predictions are estimates based on historical patterns
        - External factors (trending topics, algorithm changes) are not captured
        - Best used as a guideline, not absolute truth
        """)


if __name__ == '__main__':
    main()
