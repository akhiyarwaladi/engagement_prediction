# ROADMAP SIMPLIFIED - MVP APPROACH 🚀
# Pendekatan Pragmatis untuk Proposal Penelitian

> **Philosophy:** Start simple, iterate fast, deliver results

## 🎯 STRATEGI: 80/20 RULE

**Fokus:** 20% effort → 80% hasil

**Prinsip:**
1. ✅ Gunakan **fitur yang sudah ada** dulu
2. ✅ **One model at a time** (RF dulu, XGBoost belakangan)
3. ✅ **Skip complex features** di awal (computer vision nanti)
4. ✅ **Iterative improvement** bukan perfectionism
5. ✅ **Publikasi cepat** dengan hasil preliminary

---

## 📊 CURRENT STATE ASSESSMENT

### Yang Sudah Ada (Bisa Langsung Dipakai):
✅ Dataset 227 posts
✅ Caption text
✅ Likes count (target variable)
✅ Posting date/time
✅ Hashtags & mentions
✅ Media type (photo/video)

### Total Features Tersedia: **~12 features**
Cukup untuk baseline model yang solid!

---

## 🏃 PHASE 1: QUICK WIN (Bulan 1-3)
### Goal: Baseline model + Proof of concept

### Week 1-2: Data Preparation
```python
# Extend current dataset dengan fitur SEDERHANA
# File: src/prepare_baseline_features.py

import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv('fst_unja_from_gallery_dl.csv')

# Feature Engineering MUDAH (NO computer vision, NO Indonesian NLP yet)
df['caption_length'] = df['caption'].str.len()
df['word_count'] = df['caption'].str.split().str.len()
df['hashtag_count'] = df['hashtags_count']  # Sudah ada
df['mention_count'] = df['mentions_count']  # Sudah ada
df['is_video'] = df['is_video'].astype(int)  # Sudah ada

# Temporal features (SIMPLE)
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['month'] = df['date'].dt.month

# Engagement rate (jika follower count available - asumsi 4631 followers)
FOLLOWER_COUNT = 4631  # From screenshot
df['engagement_rate'] = df['likes'] / FOLLOWER_COUNT * 100

# Target variable
df['likes_target'] = df['likes']

# Select features
FEATURES = [
    'caption_length', 'word_count', 'hashtag_count', 'mention_count',
    'is_video', 'hour', 'day_of_week', 'is_weekend', 'month'
]

X = df[FEATURES]
y = df['likes_target']

# Save
df.to_csv('data/baseline_dataset.csv', index=False)
print(f"✅ Baseline dataset ready: {len(df)} rows, {len(FEATURES)} features")
```

**Deliverable:** `baseline_dataset.csv` dengan 9 fitur sederhana

---

### Week 3-4: Baseline Model (SIMPLE!)

```python
# File: src/baseline_model.py

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load data
df = pd.read_csv('data/baseline_dataset.csv')
X = df[FEATURES]
y = df['likes_target']

# Split (70-30 untuk awal, simple)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# MODEL: Random Forest ONLY (skip XGBoost dulu)
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# Train
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Baseline Model Performance:")
print(f"   MAE: {mae:.2f} likes")
print(f"   R²: {r2:.3f}")

# Feature Importance
importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print("\n📊 Feature Importance:")
print(importance)

# Save model
joblib.dump(rf, 'models/baseline_rf_model.pkl')
```

**Target Performance:**
- MAE < 60 likes (avg=256, so ~23% error) ✅ Good enough for baseline!
- R² > 0.50 ✅ Acceptable for social media

**Deliverable:** Working model dengan 9 fitur sederhana

---

### Week 5-8: Paper Draft 1 (QUICK!)

**Strategy:** Tulis paper SEKARANG dengan hasil baseline
- **Title:** "Prediksi Engagement Instagram untuk Institusi Akademik: Pendekatan Machine Learning Sederhana"
- **Focus:** Proof of concept bahwa ML works untuk Instagram akademik
- **Length:** 6-8 halaman

**Paper Structure:**
```markdown
1. PENDAHULUAN (1.5 hal)
   - Problem: Unpredictable Instagram engagement di FST UNJA
   - Gap: Belum ada model untuk Instagram akademik Indonesia
   - Contribution: Baseline model dengan 9 fitur sederhana

2. METODOLOGI (2 hal)
   - Dataset: 227 posts, 2022-2024
   - Features: 9 fitur (caption, hashtag, temporal)
   - Model: Random Forest
   - Evaluation: Train-test split, MAE, R²

3. HASIL (2 hal)
   - Performance metrics
   - Feature importance analysis
   - Insights (e.g., "posting time matters", "hashtags help")

4. DISKUSI (1.5 hal)
   - Interpretasi hasil
   - Limitations (small dataset, simple features, no visual analysis)
   - Future work (XGBoost, computer vision, SHAP)

5. KESIMPULAN (0.5 hal)
```

**Target Journal:** Jurnal SINTA 3-4 dulu (lebih mudah accepted)
- Contoh: Jurnal Teknologi Informasi dan Ilmu Komputer (JTIIK) - SINTA 3
- Atau: Jurnal Sistem Informasi (JSI) - SINTA 3

**Timeline:**
- Week 5-6: Tulis draft
- Week 7: Internal review
- Week 8: Submit!

**Deliverable:** 1 paper submitted ✅

---

## 🔧 PHASE 2: ENHANCEMENT (Bulan 4-6)
### Goal: Better features + Better model

### Month 4: Add Indonesian NLP (SIMPLE!)

```python
# File: src/enhanced_textual_features.py

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Setup Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def extract_enhanced_text_features(caption):
    # Stem
    caption_stemmed = stemmer.stem(caption)

    # Simple sentiment (rule-based, NO complex ML)
    positive_words = ['selamat', 'prestasi', 'juara', 'bangga', 'sukses', 'hebat']
    negative_words = ['susah', 'sulit', 'gagal']

    pos_count = sum(1 for word in positive_words if word in caption_stemmed.lower())
    neg_count = sum(1 for word in negative_words if word in caption_stemmed.lower())
    sentiment_score = pos_count - neg_count  # Simple!

    # Emoji count (simple regex)
    import re
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        "]+", flags=re.UNICODE)
    emoji_count = len(emoji_pattern.findall(caption))

    return {
        'sentiment_score': sentiment_score,
        'emoji_count': emoji_count,
        'has_question': 1 if '?' in caption else 0,
        'has_exclamation': 1 if '!' in caption else 0
    }

# Apply to dataset
for idx, row in df.iterrows():
    features = extract_enhanced_text_features(row['caption'])
    for key, val in features.items():
        df.at[idx, key] = val
```

**New Features:** +4 (total 13 features)

---

### Month 5: Add Academic Calendar (MANUAL, SIMPLE!)

```python
# File: data/academic_calendar.csv
# Buat manual di Excel, import

date,event_type,is_active_period
2022-09-01,semester_start,1
2022-10-15,midterm_exam,0
2023-01-15,final_exam,0
2023-02-10,graduation,1
2023-07-01,registration,1
# ... dst

# Merge dengan dataset
calendar = pd.read_csv('data/academic_calendar.csv')
calendar['date'] = pd.to_datetime(calendar['date'])

# Untuk setiap post, cari event terdekat
def get_academic_context(post_date, calendar):
    # Simple: check if within 7 days of any event
    for _, event in calendar.iterrows():
        days_diff = abs((post_date - event['date']).days)
        if days_diff <= 7:
            return event['event_type'], 1
    return 'regular', 0

df['event_type'], df['near_event'] = zip(*df['date'].apply(
    lambda x: get_academic_context(x, calendar)
))

# One-hot encode event_type
df = pd.get_dummies(df, columns=['event_type'], prefix='event')
```

**New Features:** +6 (total 19 features)

---

### Month 6: Improve Model (Add XGBoost Now)

```python
# File: src/improved_model.py

from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

# Load enhanced dataset (19 features)
X = df[ALL_19_FEATURES]
y = df['likes_target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest (baseline)
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# XGBoost (new)
xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)

# Simple Ensemble (average)
y_pred_rf = rf.predict(X_test)
y_pred_xgb = xgb.predict(X_test)
y_pred_ensemble = 0.5 * y_pred_rf + 0.5 * y_pred_xgb

# Compare
print("Random Forest MAE:", mean_absolute_error(y_test, y_pred_rf))
print("XGBoost MAE:", mean_absolute_error(y_test, y_pred_xgb))
print("Ensemble MAE:", mean_absolute_error(y_test, y_pred_ensemble))
```

**Expected Improvement:** MAE berkurang 10-20% dari baseline

**Deliverable:** Improved model dengan 19 fitur

---

## 📱 PHASE 3: APPLICATION & PUBLICATION (Bulan 7-12)
### Goal: Streamlit app + Finalize papers

### Month 7-9: Simple Streamlit App

```python
# File: app/simple_app.py

import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('../models/ensemble_model.pkl')

st.title("📊 Instagram Engagement Predictor")
st.write("FST UNJA - Prediksi likes untuk postingan Instagram")

# INPUT
caption = st.text_area("Caption:", placeholder="Tuliskan caption...")
hashtag_count = st.number_input("Jumlah Hashtag:", 0, 30, 5)
mention_count = st.number_input("Jumlah Mention:", 0, 10, 0)
is_video = st.checkbox("Video?")
post_date = st.date_input("Tanggal Posting:")
post_time = st.time_input("Jam Posting:")

if st.button("Prediksi"):
    # Extract features (simplified)
    features = {
        'caption_length': len(caption),
        'word_count': len(caption.split()),
        'hashtag_count': hashtag_count,
        'mention_count': mention_count,
        'is_video': int(is_video),
        'hour': post_time.hour,
        'day_of_week': post_date.weekday(),
        'is_weekend': int(post_date.weekday() >= 5),
        'month': post_date.month,
        # ... other 10 features dengan default values
    }

    X_input = pd.DataFrame([features])
    prediction = model.predict(X_input)[0]

    st.success(f"✅ Prediksi Likes: **{prediction:.0f}**")

    # Simple recommendation
    st.info("💡 Tips: Posting di jam 10-12 siang biasanya mendapat lebih banyak engagement!")
```

**NO SHAP dulu, NO complex visualizations** - Keep it simple!

**Deliverable:** Working web app

---

### Month 10: SHAP Analysis (Optional)

```python
# File: src/shap_simple.py

import shap
import matplotlib.pyplot as plt

# SHAP untuk XGBoost only (lebih mudah)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Summary plot (global)
shap.summary_plot(shap_values, X_test, feature_names=FEATURES, show=False)
plt.savefig('docs/shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# Bar plot (feature importance)
shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
plt.savefig('docs/shap_importance.png', dpi=300, bbox_inches='tight')
```

Cukup 2 plot ini untuk paper!

---

### Month 11-12: Finalize Papers

**Paper 2** (Target: SINTA 2)
- Title: "Enhanced Instagram Engagement Prediction dengan Ensemble Learning dan Feature Engineering"
- Content: Full model dengan 19 fitur + XGBoost + SHAP
- Target: Jurnal SINTA 2 (lebih ambisius)

**Conference Paper**
- Present preliminary results di Seminar Nasional Informatika
- Format: 6 halaman, fokus pada aplikasi praktis

**Policy Brief**
- 2 halaman untuk Tim Humas FST
- Rekomendasi: optimal posting time, hashtag strategy, content type

---

## 📦 TECH STACK (MINIMAL)

```txt
# requirements_minimal.txt

pandas
numpy
scikit-learn
xgboost
Sastrawi
matplotlib
seaborn
streamlit
joblib
openpyxl  # untuk Excel calendar
```

**That's it!** No OpenCV, no complex CV, no NLTK, no heavy dependencies

---

## ⏱️ TIME INVESTMENT

**Per Week:**
- Week 1-12 (Phase 1): ~15 jam/minggu
- Week 13-24 (Phase 2): ~12 jam/minggu
- Week 25-48 (Phase 3): ~10 jam/minggu

**Total:** ~600 jam vs 1200+ jam di roadmap original (50% less work!)

---

## ✅ SUCCESS CRITERIA (REALISTIC)

### Minimum Viable Results:
- ✅ MAE < 70 likes (~27% error) ← More achievable
- ✅ R² > 0.50 ← Acceptable untuk social media
- ✅ 2 papers submitted (1 bisa SINTA 3-4) ← More realistic
- ✅ 1 conference paper accepted
- ✅ 1 working Streamlit app

### Stretch Goals (If time permits):
- 🎯 MAE < 50 likes
- 🎯 R² > 0.65
- 🎯 SHAP interpretability
- 🎯 Computer vision features

---

## 🚨 WHEN TO STOP & SHIP

**Red Flags untuk Over-engineering:**
- ❌ Spending >2 weeks on feature engineering yang gain <5% performance
- ❌ Trying to implement ALL 55 features dari proposal
- ❌ Perfectionism pada Streamlit UI
- ❌ Waiting for "perfect" results untuk submit paper

**Golden Rule:**
> "Done is better than perfect. Ship early, iterate based on feedback."

---

## 🎯 NEXT ACTIONS (THIS WEEK!)

### Day 1-2:
```bash
# 1. Setup simple structure
mkdir -p data/raw data/processed models app docs

# 2. Run baseline feature extraction
python src/prepare_baseline_features.py
```

### Day 3-4:
```bash
# 3. Train baseline model
python src/baseline_model.py
```

### Day 5:
```bash
# 4. Evaluate results
# If MAE < 70 likes → GOOD! Proceed to paper
# If MAE > 100 likes → Debug, add more features
```

### Day 6-7:
```bash
# 5. Start paper draft outline
# Just outline, bullet points
```

---

## 💡 KEY INSIGHTS

### Why This Approach Works:
1. **Fast Validation:** Tahu dalam 2 minggu apakah approach viable
2. **Early Wins:** Paper 1 submitted di bulan 3 → momentum!
3. **Iterative:** Bisa adjust based on results
4. **Low Risk:** Tidak invest terlalu banyak di awal
5. **Publication Priority:** Papers > Perfect model

### What We're Skipping (For Now):
- ❌ Complex computer vision (save 100+ hours)
- ❌ Advanced Indonesian NLP (save 50+ hours)
- ❌ Hyperparameter tuning dengan GridSearch (save 20+ hours)
- ❌ Complex SHAP for every prediction (save 30+ hours)
- ❌ Production-grade deployment (save 40+ hours)

**Total Time Saved: 240+ hours!**

---

## 📚 COMPARISON: Original vs Simplified

| Aspect | Original Roadmap | Simplified (MVP) |
|--------|------------------|-------------------|
| **Features** | 55 (text + visual + temporal) | 9 → 19 (text + temporal only) |
| **Models** | XGBoost + RF + Ensemble optimization | RF → Simple ensemble |
| **Time to First Result** | 6 months | 3 weeks |
| **Time to First Paper** | 9 months | 3 months |
| **Complexity** | High (Computer Vision, Advanced NLP) | Medium (Basic NLP, no CV) |
| **Success Rate** | 60% (high risk) | 85% (low risk) |
| **Papers** | 3 (all SINTA 2) | 2-3 (mix SINTA 2-4) |

**Verdict:** Simplified approach = 70% hasil dengan 50% effort ✅

---

## 🎓 PHILOSOPHY

> "Research is iterative. Start with the simplest thing that could possibly work, then improve based on evidence."

**Remember:**
- Baseline model dengan 9 fitur bisa jadi cukup powerful!
- Most of prediction power often comes from <10 core features
- Instagram engagement is inherently noisy (R²=0.50 is OK!)
- Publications value novelty & insights, not just SOTA performance

---

## ✨ BONUS: Quick Wins Checklist

Week 1:
- [ ] Create `data/baseline_dataset.csv`
- [ ] Train Random Forest baseline
- [ ] Get MAE & R²

Week 2:
- [ ] Feature importance analysis
- [ ] Identify top 5 most important features
- [ ] Visualization (actual vs predicted)

Week 3:
- [ ] Start paper draft outline
- [ ] Introduction + Methodology sections

Week 4:
- [ ] Results section with plots
- [ ] Draft discussion

Month 2:
- [ ] Internal review
- [ ] Revise based on feedback

Month 3:
- [ ] Submit Paper 1! 🎉

---

**Want to start? Run this first:**

```bash
# Validate current data
python -c "
import pandas as pd
df = pd.read_csv('fst_unja_from_gallery_dl.csv')
print(f'Dataset: {len(df)} posts')
print(f'Avg likes: {df[\"likes\"].mean():.2f}')
print(f'Features: {df.columns.tolist()}')
print(f'✅ Ready untuk feature engineering!')
"
```

---

**Last Updated:** 2025-10-02
**Version:** 1.0 - MVP Approach
**Status:** Ready to implement 🚀
