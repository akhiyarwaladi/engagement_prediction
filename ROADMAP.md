# ROADMAP IMPLEMENTASI PENELITIAN
# Prediksi Engagement Instagram FST UNJA

## 📋 Executive Summary

**Objective:** Mengembangkan model prediksi engagement Instagram menggunakan ensemble machine learning (XGBoost + Random Forest) dengan 55 fitur multimodal

**Timeline:** 12 bulan (September 2025 - Agustus 2026)
**Budget:** Rp 50.000.000
**Dataset:** 227 posts Instagram @fst_unja (2022-2024)
**Target Output:** Model ML + Aplikasi Streamlit + 3 Publikasi

---

## 🎯 FASE 1: PERSIAPAN & DATA ENHANCEMENT (Bulan 1-3)

### Month 1: Setup & Academic Calendar

**Week 1-2: Project Setup**
```bash
# Reorganisasi struktur project
mkdir -p {data/{raw,processed,features,academic},src/{collection,preprocessing,features,models,evaluation,utils},app,notebooks,tests,docs,models}

# Install dependencies
pip install pandas numpy scikit-learn xgboost sastrawi opencv-python pillow matplotlib seaborn shap streamlit tqdm pyyaml
```

**Week 3-4: Academic Calendar Database**
- [ ] Buat `data/academic/calendar_unja_2022_2024.csv` dengan kolom:
  - date, event_type (ujian/wisuda/pendaftaran/libur/kegiatan), semester, is_academic_period
- [ ] Script `src/collection/calendar_mapper.py` untuk map posts ke periode akademik
- [ ] Add academic context flags ke dataset

**Deliverable:** Dataset enhanced dengan 3 fitur temporal baru (academic_period, event_proximity, semester)

---

### Month 2: Complete Data Collection

**Visual Data Download**
- [ ] Script `src/collection/image_downloader.py` untuk download semua gambar dari gallery-dl results
- [ ] Organize images di `data/raw/images/{post_id}.jpg`
- [ ] Validate: pastikan 227 images ter-download

**Enhanced Metadata**
- [ ] Coba akses Instagram Business API untuk comments/shares (jika memungkinkan)
- [ ] Document limitations (comments=0 dari gallery-dl)
- [ ] Create data quality report

**Deliverable:** Complete dataset dengan gambar + metadata lengkap

---

### Month 3: Exploratory Data Analysis

**Statistical Analysis**
- [ ] Jupyter notebook `notebooks/01_eda.ipynb`:
  - Distribution analysis (likes, posting time, content type)
  - Correlation analysis
  - Temporal patterns (day of week, hour, academic period)
  - Content type analysis (photo vs video)
  - Hashtag effectiveness analysis

**Deliverable:** EDA report dengan insights untuk feature engineering

---

## 🔧 FASE 2: FEATURE ENGINEERING (Bulan 4-6)

### Month 4: Textual Features (20 features)

**Basic Text Features (5 features)**
```python
# src/features/textual_features.py

def extract_basic_text_features(caption):
    return {
        'caption_length': len(caption),
        'word_count': len(caption.split()),
        'sentence_count': caption.count('.') + caption.count('!') + caption.count('?'),
        'avg_word_length': np.mean([len(w) for w in caption.split()]),
        'char_count': len(caption)
    }
```

**Linguistic Features (6 features)**
- [ ] Emoji count & types
- [ ] Mention count (@username)
- [ ] URL presence
- [ ] Question mark presence
- [ ] Exclamation mark count
- [ ] Uppercase word ratio

**Sentiment Analysis (2 features)**
- [ ] Install/adapt Indonesian sentiment analyzer
- [ ] Polarity score (-1 to 1)
- [ ] Subjectivity score (0 to 1)

**Hashtag Features (5 features)**
- [ ] Hashtag count
- [ ] Academic hashtag ratio (#wisuda, #fst, #unja)
- [ ] Popular hashtag ratio (top 20 from historical data)
- [ ] Hashtag diversity (unique/total)
- [ ] Average hashtag length

**TF-IDF Features (7 features - top topics)**
- [ ] Vectorize all captions
- [ ] Extract top 7 topic components
- [ ] Save vocabulary for inference

**Deliverable:** `data/features/textual_features.csv` dengan 20 kolom

---

### Month 5: Visual Features (20 features)

**Color Histogram (12 features)**
```python
# src/features/visual_features.py
import cv2
import numpy as np

def extract_color_features(image_path):
    img = cv2.imread(image_path)

    # RGB mean & std (6 features)
    rgb_mean = img.mean(axis=(0,1))  # R, G, B mean
    rgb_std = img.std(axis=(0,1))    # R, G, B std

    # HSV mean & std (6 features)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_mean = hsv.mean(axis=(0,1))  # H, S, V mean
    hsv_std = hsv.std(axis=(0,1))    # H, S, V std

    return np.concatenate([rgb_mean, rgb_std, hsv_mean, hsv_std])
```

**Face Detection (2 features)**
- [ ] Haar Cascade implementation
- [ ] Face count
- [ ] Face presence (binary flag)

**Image Composition (4 features)**
- [ ] Brightness (mean luminance)
- [ ] Contrast (std luminance)
- [ ] Saturation (mean from HSV)
- [ ] Edge density (Canny edge detection)

**Format Features (2 features)**
- [ ] Media type (photo=0, video=1)
- [ ] Aspect ratio (width/height)

**Deliverable:** `data/features/visual_features.csv` dengan 20 kolom

---

### Month 6: Temporal Features & Integration (15 features)

**Time Features (9 features)**
```python
# src/features/temporal_features.py

def extract_temporal_features(timestamp):
    dt = pd.to_datetime(timestamp)

    # Cyclic encoding untuk hour (2 features)
    hour_sin = np.sin(2 * np.pi * dt.hour / 24)
    hour_cos = np.cos(2 * np.pi * dt.hour / 24)

    # One-hot encoding untuk day of week (7 features)
    day_onehot = [1 if i == dt.dayofweek else 0 for i in range(7)]

    return [hour_sin, hour_cos] + day_onehot
```

**Academic Context (6 features)**
- [ ] Is exam period (binary)
- [ ] Is registration period (binary)
- [ ] Is graduation period (binary)
- [ ] Is semester break (binary)
- [ ] Days to next major event
- [ ] Semester indicator (1 or 2)

**Feature Integration**
- [ ] Combine all features: `src/features/feature_combiner.py`
- [ ] Handle missing values
- [ ] Feature scaling/normalization
- [ ] Save final feature matrix: `data/features/final_dataset.csv` (227 rows × 55 features)

**Deliverable:** Complete dataset dengan 55 fitur siap untuk ML

---

## 🤖 FASE 3: MODEL DEVELOPMENT (Bulan 7-8)

### Month 7: Individual Models

**Data Splitting**
```python
# src/models/train_test_split.py

from sklearn.model_selection import train_test_split, StratifiedKFold

# Categorize engagement untuk stratification
y_categories = pd.cut(y_likes, bins=3, labels=['low', 'medium', 'high'])

# 70-15-15 split dengan stratification
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y_categories, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp_categories, random_state=42
)

# Result: 159 train, 34 val, 34 test
```

**XGBoost Model**
```python
# src/models/xgboost_model.py

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

xgb = XGBRegressor(random_state=42)
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

best_xgb = grid_search.best_estimator_
```

**Random Forest Model**
```python
# src/models/random_forest_model.py

from sklearn.ensemble import RandomForestRegressor

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 8, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5]
}

rf = RandomForestRegressor(random_state=42)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='neg_mean_absolute_error')
grid_search_rf.fit(X_train, y_train)

best_rf = grid_search_rf.best_estimator_
```

**Deliverable:** Trained XGBoost & RF models disimpan di `models/`

---

### Month 8: Ensemble & Optimization

**Weighted Ensemble**
```python
# src/models/ensemble.py

class WeightedEnsemble:
    def __init__(self, xgb_model, rf_model, xgb_weight=0.6, rf_weight=0.4):
        self.xgb = xgb_model
        self.rf = rf_model
        self.w1 = xgb_weight
        self.w2 = rf_weight

    def predict(self, X):
        pred_xgb = self.xgb.predict(X)
        pred_rf = self.rf.predict(X)
        return self.w1 * pred_xgb + self.w2 * pred_rf

# Find optimal weights using validation set
from scipy.optimize import minimize

def objective(weights):
    ensemble = WeightedEnsemble(xgb, rf, weights[0], weights[1])
    pred = ensemble.predict(X_val)
    return mean_absolute_error(y_val, pred)

result = minimize(objective, [0.5, 0.5], bounds=[(0,1), (0,1)],
                  constraints={'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1})
optimal_weights = result.x
```

**5-Fold Cross Validation**
```python
# src/evaluation/cross_validation.py

from sklearn.model_selection import cross_val_score

cv_scores_xgb = cross_val_score(xgb, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')

print(f"XGBoost CV MAE: {-cv_scores_xgb.mean():.2f} ± {cv_scores_xgb.std():.2f}")
print(f"Random Forest CV MAE: {-cv_scores_rf.mean():.2f} ± {cv_scores_rf.std():.2f}")
```

**Deliverable:** Optimized ensemble model dengan bobot optimal

---

## 📊 FASE 4: EVALUATION & INTERPRETABILITY (Bulan 9-10)

### Month 9: Comprehensive Evaluation

**Classification Metrics** (engagement categories: low/medium/high)
```python
# src/evaluation/metrics.py

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Convert regression to classification untuk evaluation
y_test_cat = pd.cut(y_test, bins=3, labels=[0, 1, 2])
y_pred_cat = pd.cut(ensemble.predict(X_test), bins=3, labels=[0, 1, 2])

accuracy = accuracy_score(y_test_cat, y_pred_cat)
precision, recall, f1, _ = precision_recall_fscore_support(y_test_cat, y_pred_cat, average='macro')

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")
```

**Regression Metrics**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = ensemble.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f} likes")
print(f"RMSE: {rmse:.2f} likes")
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2:.3f}")
```

**Visualizations**
- [ ] Confusion matrix heatmap
- [ ] Actual vs Predicted scatter plot
- [ ] Residual plot
- [ ] Learning curves
- [ ] Per-class performance

**Deliverable:** Comprehensive evaluation report

---

### Month 10: Interpretability & Feature Importance

**SHAP Analysis**
```python
# src/evaluation/shap_analysis.py

import shap

# Create SHAP explainer
explainer_xgb = shap.TreeExplainer(xgb_model)
shap_values_xgb = explainer_xgb.shap_values(X_test)

# Global feature importance
shap.summary_plot(shap_values_xgb, X_test, feature_names=feature_names, show=False)
plt.savefig('docs/figures/shap_summary.png', dpi=300, bbox_inches='tight')

# Top 10 most important features
shap.summary_plot(shap_values_xgb, X_test, feature_names=feature_names,
                  plot_type='bar', max_display=10, show=False)
plt.savefig('docs/figures/shap_top10.png', dpi=300, bbox_inches='tight')

# Local explanations for specific predictions
for i in [0, 10, 20]:  # Sample posts
    shap.force_plot(explainer_xgb.expected_value, shap_values_xgb[i],
                    X_test.iloc[i], feature_names=feature_names,
                    matplotlib=True, show=False)
    plt.savefig(f'docs/figures/shap_local_{i}.png', dpi=300, bbox_inches='tight')
```

**Feature Importance Comparison**
```python
# Compare XGBoost vs Random Forest feature importance

import pandas as pd

# XGBoost feature importance
xgb_importance = pd.DataFrame({
    'feature': feature_names,
    'importance_xgb': xgb_model.feature_importances_
}).sort_values('importance_xgb', ascending=False)

# Random Forest feature importance
rf_importance = pd.DataFrame({
    'feature': feature_names,
    'importance_rf': rf_model.feature_importances_
}).sort_values('importance_rf', ascending=False)

# Merge and compare
importance_df = xgb_importance.merge(rf_importance, on='feature')
importance_df.to_csv('docs/feature_importance_comparison.csv', index=False)
```

**Temporal Analysis**
- [ ] Performance by academic period
- [ ] Performance by day of week
- [ ] Performance by semester

**Deliverable:** Interpretability report dengan SHAP visualizations

---

## 💻 FASE 5: APPLICATION DEVELOPMENT (Bulan 11-12)

### Month 11: Streamlit Web Application

**App Structure**
```python
# app/streamlit_app.py

import streamlit as st
import pickle
from PIL import Image

# Load trained model
@st.cache_resource
def load_model():
    with open('../models/ensemble_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("📊 Instagram Engagement Predictor")
st.subheader("FST UNJA Social Media Analytics")

# Tabs
tab1, tab2, tab3 = st.tabs(["Predict", "Analytics", "About"])

with tab1:
    st.header("Predict Post Engagement")

    # Input form
    caption = st.text_area("Caption", placeholder="Masukkan caption postingan...")
    uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    col1, col2 = st.columns(2)
    with col1:
        post_date = st.date_input("Post Date")
        post_time = st.time_input("Post Time")
    with col2:
        is_exam = st.checkbox("During Exam Period")
        is_graduation = st.checkbox("Near Graduation")

    if st.button("Predict Engagement"):
        # Extract features
        features = extract_all_features(caption, uploaded_image, post_date, post_time,
                                       is_exam, is_graduation)

        # Predict
        predicted_likes = model.predict([features])[0]

        # Display results
        st.success(f"**Predicted Likes: {predicted_likes:.0f}**")

        # SHAP explanation
        st.subheader("Why this prediction?")
        shap_explanation = explain_prediction(model, features)
        st.pyplot(shap_explanation)

        # Recommendations
        st.subheader("Recommendations")
        recommendations = generate_recommendations(features, predicted_likes)
        for rec in recommendations:
            st.info(rec)

with tab2:
    st.header("Historical Performance")

    # Load historical data
    df = pd.read_csv('../data/final_dataset.csv')

    # Engagement trend over time
    fig_trend = px.line(df, x='date', y='likes', title='Engagement Trend Over Time')
    st.plotly_chart(fig_trend)

    # Top performing posts
    st.subheader("Top 10 Posts")
    top_posts = df.nlargest(10, 'likes')[['date', 'caption', 'likes', 'media_type']]
    st.dataframe(top_posts)

    # Performance by content type
    fig_type = px.box(df, x='media_type', y='likes', title='Engagement by Content Type')
    st.plotly_chart(fig_type)

    # Performance by posting time
    df['hour'] = pd.to_datetime(df['date']).dt.hour
    fig_time = px.bar(df.groupby('hour')['likes'].mean().reset_index(),
                     x='hour', y='likes', title='Average Engagement by Hour')
    st.plotly_chart(fig_time)

with tab3:
    st.header("About This Tool")
    st.markdown("""
    ### Model Information
    - **Algorithm**: Ensemble (XGBoost + Random Forest)
    - **Features**: 55 multimodal features (text, visual, temporal)
    - **Training Data**: 227 Instagram posts from @fst_unja (2022-2024)
    - **Performance**: MAE = X.XX likes, R² = 0.XX

    ### Team
    - **Principal Investigator**: Jefri Marzal
    - **Research Team**: Muhammad Razi A., Miranty Yudistira, Akhiyar Waladi, Hamzah Alghifari

    ### Citation
    *Paper under review*
    """)
```

**Features to Implement:**
- [ ] Caption input dengan preview
- [ ] Image upload dengan preview
- [ ] DateTime picker dengan academic calendar hints
- [ ] Prediction dengan confidence interval
- [ ] SHAP local explanation untuk setiap prediksi
- [ ] Recommendations (optimal posting time, hashtag suggestions, etc.)
- [ ] Historical analytics dashboard
- [ ] Model performance metrics

**Deployment:**
```bash
# Local testing
streamlit run app/streamlit_app.py

# Deploy ke Streamlit Cloud
# 1. Push to GitHub
# 2. Connect to Streamlit Cloud
# 3. Deploy
```

**Deliverable:** Functional web application

---

### Month 12: Documentation & Dissemination

**Technical Documentation**
- [ ] `docs/methodology.md`: Detailed methodology
- [ ] `docs/api_documentation.md`: API usage guide
- [ ] `docs/user_guide.md`: End-user manual
- [ ] `docs/model_card.md`: Model specifications & limitations

**Academic Papers**
- [ ] **Paper 1** (Target: Jurnal SINTA 2):
  - Title: "Prediksi Engagement Instagram Menggunakan Ensemble Learning untuk Institusi Akademik"
  - Focus: Methodology + Results

- [ ] **Paper 2** (Target: Jurnal SINTA 2):
  - Title: "Feature Engineering Multimodal untuk Prediksi Media Sosial Konteks Indonesia"
  - Focus: Feature engineering techniques

- [ ] **Conference Paper** (Target: Seminar Nasional):
  - Title: "Aplikasi Machine Learning untuk Optimasi Strategi Konten Media Sosial Perguruan Tinggi"
  - Focus: Practical application

**Policy Brief**
- [ ] Draft policy brief untuk FST UNJA
- [ ] Recommendations untuk content strategy
- [ ] Best practices untuk Instagram engagement

**Final Report**
- [ ] Executive summary
- [ ] Detailed methodology
- [ ] Results & analysis
- [ ] Application user guide
- [ ] Future work recommendations

**Deliverable:** Complete documentation + 3 publications submitted

---

## 📦 DEPENDENCIES & TOOLS

### Python Libraries

```txt
# requirements.txt

# Core Data Science
pandas==2.1.0
numpy==1.24.3
scipy==1.11.2

# Machine Learning
scikit-learn==1.3.0
xgboost==2.0.0
imbalanced-learn==0.11.0  # For SMOTE

# Indonesian NLP
Sastrawi==1.2.0
nltk==3.8.1

# Computer Vision
opencv-python==4.8.0
Pillow==10.0.0
scikit-image==0.21.0

# Interpretability
shap==0.42.1
lime==0.2.0.1  # Alternative interpretability

# Visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.16.1

# Web Application
streamlit==1.26.0
fastapi==0.103.0  # For API
uvicorn==0.23.2

# Data Collection
gallery-dl==1.25.8
instaloader==4.10  # Alternative scraper

# Utilities
python-dotenv==1.0.0
pyyaml==6.0.1
tqdm==4.66.1
joblib==1.3.2

# Testing
pytest==7.4.0
pytest-cov==4.1.0
```

### Development Tools

- **IDE**: VS Code dengan Python extension
- **Version Control**: Git + GitHub
- **Notebook**: Jupyter Lab
- **Documentation**: Markdown + Sphinx
- **Deployment**: Streamlit Cloud (free tier)

---

## ⚠️ CHALLENGES & SOLUTIONS

### Challenge 1: Dataset Kecil (227 posts)

**Problem:** Overfitting risk dengan 55 features

**Solutions:**
1. **Regularization**: L1/L2 dalam model
2. **Feature Selection**: Remove low-importance features
3. **Cross-Validation**: 5-fold untuk robust evaluation
4. **Ensemble Methods**: Reduce variance
5. **SMOTE (optional)**: Balance classes jika perlu

```python
# Feature selection untuk reduce overfitting
from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(f_regression, k=30)  # Keep top 30 features
X_selected = selector.fit_transform(X_train, y_train)
```

### Challenge 2: Indonesian Language Processing

**Problem:** Limited NLP tools untuk bahasa Indonesia akademik

**Solutions:**
1. **Sastrawi**: Use for stemming
2. **Custom Dictionary**: Build academic term dictionary
3. **Sentiment Lexicon**: Adapt for Indonesian context
4. **Manual Annotation**: Create small labeled dataset for sentiment

```python
# Custom academic dictionary
ACADEMIC_TERMS = {
    'wisuda': 'graduation',
    'ujian': 'exam',
    'mahasiswa': 'student',
    'dosen': 'lecturer',
    # ... add more
}

# Preserve academic terms during preprocessing
def preprocess_academic_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())

    # Stem non-academic terms only
    stemmed = []
    for token in tokens:
        if token in ACADEMIC_TERMS:
            stemmed.append(token)  # Preserve
        else:
            stemmed.append(stemmer.stem(token))

    return ' '.join(stemmed)
```

### Challenge 3: Missing Comments Data

**Problem:** gallery-dl tidak provide comments count

**Solutions:**
1. **Primary Metric**: Focus on likes sebagai main engagement indicator
2. **Document Limitation**: Explain dalam paper
3. **Future Work**: Instagram Business API integration untuk comments
4. **Alternative Metrics**: Engagement rate = likes / followers (jika follower count available)

### Challenge 4: Visual Feature Extraction Complexity

**Problem:** Complex computer vision dengan dataset kecil

**Solutions:**
1. **Start Simple**: Color histogram, face detection (basic features)
2. **Pre-trained Models**: Use VGG/ResNet features jika perlu
3. **Transfer Learning**: Extract features from pre-trained CNN
4. **Manual Validation**: Validate visual features make sense

```python
# Transfer learning approach (if needed)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

def extract_vgg_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = vgg_model.predict(x)
    return features.flatten()
```

### Challenge 5: Real-time Deployment

**Problem:** Model inference speed untuk production

**Solutions:**
1. **Model Optimization**: Prune trees, reduce estimators
2. **Caching**: Cache feature extraction results
3. **Async Processing**: FastAPI untuk non-blocking inference
4. **Quantization**: Reduce model size jika perlu

---

## 🎯 SUCCESS CRITERIA

### Technical Metrics

- **MAE < 50 likes** (avg post = 256 likes, so ~20% error)
- **R² > 0.60** (reasonable for social media prediction)
- **F1-Score > 0.70** (untuk classification task)
- **Cross-validation stability**: CV std < 20% of mean

### Research Outputs

- [x] 3 Journal papers (2 submitted, 1 accepted) - **Mandatory**
- [x] 1 Conference paper (accepted) - **Mandatory**
- [x] 1 Policy brief (published) - **Mandatory**
- [x] 1 Web application (deployed) - **Target**
- [x] 1 Open dataset (published) - **Bonus**

### Impact Metrics

- **Adoption**: Tim Humas FST UNJA menggunakan tool
- **Engagement Improvement**: Avg likes meningkat 15%+ setelah 6 bulan
- **Time Efficiency**: Reduce content planning time 30%

---

## 📚 LEARNING RESOURCES

### Machine Learning
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [SHAP Documentation](https://shap.readthedocs.io/)

### Indonesian NLP
- [Sastrawi GitHub](https://github.com/sastrawi/sastrawi)
- [Indonesian NLP Resources](https://github.com/topics/indonesian-nlp)

### Computer Vision
- [OpenCV Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
- [Image Feature Extraction](https://scikit-image.org/docs/stable/auto_examples/)

### Streamlit
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)

---

## 🔄 NEXT STEPS

**Immediate Actions (This Week):**
1. Create academic calendar CSV
2. Setup project structure
3. Install all dependencies
4. Download all images from gallery-dl
5. Start EDA notebook

**Review Schedule:**
- **Monthly**: Team meeting untuk progress update
- **Quarterly**: Evaluation dengan Tim Humas FST
- **Milestone**: After each phase completion

---

## 📞 CONTACT & SUPPORT

**Principal Investigator:**
- Jefri Marzal (Ketua) - jefri.marzal@unja.ac.id

**Team Members:**
- Muhammad Razi A. - Data Collection
- Miranty Yudistira - Text Analysis
- Akhiyar Waladi - Visual Analysis
- Hamzah Alghifari - Evaluation & App Dev

**Repository:** [GitHub Link TBD]
**Documentation:** [Docs Link TBD]

---

**Last Updated:** 2025-10-02
**Version:** 1.0
