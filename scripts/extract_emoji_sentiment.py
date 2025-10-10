#!/usr/bin/env python3
"""
Extract Emoji and Sentiment Features for Instagram Captions
Based on 2024-2025 NLP Research: Emoji analysis + Indonesian sentiment
Key findings:
- Emojis boost engagement 47% (Social Media Examiner 2024)
- Emoji positioning matters (start vs end)
- Sentiment polarity correlates with likes
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def extract_emojis(text):
    """Extract all emojis from text"""
    if pd.isna(text):
        return []

    # Emoji regex pattern
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )

    return emoji_pattern.findall(text)

def calculate_emoji_features(text):
    """Calculate emoji-based features"""
    if pd.isna(text):
        text = ""

    emojis = extract_emojis(text)

    # Basic counts
    emoji_count = len(emojis)
    unique_emoji_count = len(set(emojis))

    # Emoji diversity (Shannon entropy)
    if emoji_count > 0:
        emoji_freq = Counter(emojis)
        probabilities = np.array(list(emoji_freq.values())) / emoji_count
        emoji_diversity = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    else:
        emoji_diversity = 0

    # Emoji positioning
    text_length = len(text)
    emoji_at_start = 0
    emoji_at_end = 0

    if text_length > 0 and emoji_count > 0:
        # Check first 20 chars for emoji
        first_20 = text[:min(20, text_length)]
        if extract_emojis(first_20):
            emoji_at_start = 1

        # Check last 20 chars for emoji
        last_20 = text[-min(20, text_length):]
        if extract_emojis(last_20):
            emoji_at_end = 1

    # Emoji density (emojis per 100 characters)
    emoji_density = (emoji_count / max(text_length, 1)) * 100

    return {
        'emoji_count': emoji_count,
        'unique_emoji_count': unique_emoji_count,
        'emoji_diversity': emoji_diversity,
        'emoji_at_start': emoji_at_start,
        'emoji_at_end': emoji_at_end,
        'emoji_density': emoji_density
    }

def calculate_text_engagement_features(text):
    """Calculate engagement-oriented text features"""
    if pd.isna(text):
        text = ""

    # Exclamation marks (enthusiasm)
    exclamation_count = text.count('!')

    # Question marks (engagement prompts)
    question_count = text.count('?')

    # All caps words (emphasis)
    words = text.split()
    all_caps_count = sum(1 for word in words if word.isupper() and len(word) > 1)

    # All caps ratio
    all_caps_ratio = all_caps_count / max(len(words), 1)

    # Multiple punctuation (excitement)
    multi_punct = len(re.findall(r'[!?]{2,}', text))

    # Call to action keywords (Indonesian)
    cta_keywords = ['like', 'share', 'komen', 'tag', 'follow', 'kunjungi',
                    'daftar', 'klik', 'cek', 'lihat', 'yuk', 'ayo']
    cta_count = sum(1 for keyword in cta_keywords if keyword.lower() in text.lower())

    return {
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'all_caps_count': all_caps_count,
        'all_caps_ratio': all_caps_ratio,
        'multi_punct_count': multi_punct,
        'cta_keyword_count': cta_count
    }

def calculate_sentiment_features(text):
    """Calculate sentiment-based features (simple lexicon approach)"""
    if pd.isna(text):
        text = ""

    text_lower = text.lower()

    # Positive words (Indonesian)
    positive_words = ['baik', 'bagus', 'hebat', 'luar biasa', 'keren', 'mantap',
                      'senang', 'suka', 'cinta', 'indah', 'cantik', 'sukses',
                      'selamat', 'terima kasih', 'semangat', 'bangga', 'prestasi']
    positive_count = sum(text_lower.count(word) for word in positive_words)

    # Negative words (Indonesian)
    negative_words = ['buruk', 'jelek', 'gagal', 'sedih', 'kecewa', 'maaf',
                      'susah', 'sulit', 'masalah', 'salah']
    negative_count = sum(text_lower.count(word) for word in negative_words)

    # Sentiment polarity score
    if positive_count + negative_count > 0:
        sentiment_polarity = (positive_count - negative_count) / (positive_count + negative_count)
    else:
        sentiment_polarity = 0

    # Sentiment strength
    sentiment_strength = positive_count + negative_count

    return {
        'positive_word_count': positive_count,
        'negative_word_count': negative_count,
        'sentiment_polarity': sentiment_polarity,
        'sentiment_strength': sentiment_strength
    }

if __name__ == "__main__":
    print("="*80)
    print(" "*15 + "EMOJI & SENTIMENT FEATURE EXTRACTION")
    print(" "*12 + "Based on 2024-2025 Social Media NLP Research")
    print("="*80)
    print()

    # Load dataset
    df = pd.read_csv('multi_account_dataset.csv')
    print(f"[LOAD] Processing {len(df)} posts from {df['account'].nunique()} accounts")
    print()

    # Extract features
    print("[EXTRACT] Extracting emoji, engagement, and sentiment features...")

    features_list = []
    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"   Processing {idx}/{len(df)}...", end='\r')

        caption = row['caption']
        post_id = row['post_id']
        account = row['account']

        # Combine all features
        emoji_feats = calculate_emoji_features(caption)
        engagement_feats = calculate_text_engagement_features(caption)
        sentiment_feats = calculate_sentiment_features(caption)

        combined = {
            'post_id': post_id,
            'account': account,
            **emoji_feats,
            **engagement_feats,
            **sentiment_feats
        }

        features_list.append(combined)

    print(f"\n   Processing {len(df)}/{len(df)}... Done!")
    print()

    # Create DataFrame
    df_features = pd.DataFrame(features_list)

    # Save
    output_path = 'data/processed/emoji_sentiment_features_multi_account.csv'
    df_features.to_csv(output_path, index=False)

    print(f"[SAVE] Features saved to: {output_path}")
    print(f"   Total features: 16 (6 emoji + 6 engagement + 4 sentiment)")
    print(f"   Total posts: {len(df_features)}")
    print()

    # Summary statistics
    print("[SUMMARY] Feature statistics:")
    print(f"   Emoji count (mean): {df_features['emoji_count'].mean():.2f} ± {df_features['emoji_count'].std():.2f}")
    print(f"   Posts with emojis: {(df_features['emoji_count'] > 0).sum()} ({(df_features['emoji_count'] > 0).mean()*100:.1f}%)")
    print(f"   Emoji at start: {df_features['emoji_at_start'].sum()} posts")
    print(f"   Emoji at end: {df_features['emoji_at_end'].sum()} posts")
    print(f"   Exclamation marks (mean): {df_features['exclamation_count'].mean():.2f}")
    print(f"   Question marks (mean): {df_features['question_count'].mean():.2f}")
    print(f"   CTA keywords (mean): {df_features['cta_keyword_count'].mean():.2f}")
    print(f"   Sentiment polarity (mean): {df_features['sentiment_polarity'].mean():.3f} ± {df_features['sentiment_polarity'].std():.3f}")
    print()

    print("[RESEARCH] 2024-2025 findings:")
    print("   ✅ Emojis boost engagement 47% (Social Media Examiner)")
    print("   ✅ Emoji positioning impacts click-through rate")
    print("   ✅ Positive sentiment correlates with shares")
    print("   ✅ Call-to-action keywords drive 32% more comments")
