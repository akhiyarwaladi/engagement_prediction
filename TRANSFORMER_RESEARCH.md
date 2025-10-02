# 🤖 TRANSFORMER-BASED APPROACH - RESEARCH & IMPLEMENTATION PLAN

**Date:** October 2, 2025
**Purpose:** Phase 4 Enhancement with Pretrained Transformers
**Expected Improvement:** +0.15-0.30 R² (target R² = 0.35-0.50)

---

## 📊 EXECUTIVE SUMMARY

### Why Transformers?

Current performance (Phase 2):
- **MAE:** 109.42 likes
- **R²:** 0.200
- **Features:** Simple word-based NLP (word lists, emoji count)

**Limitation:** Cannot capture:
- ❌ Semantic meaning & context
- ❌ Complex sentiment (sarcasm, negation)
- ❌ Deep visual understanding
- ❌ Text-image relationship

**Solution:** Transformer-based deep learning
- ✅ Contextual embeddings (BERT, IndoBERT)
- ✅ Vision transformers (ViT)
- ✅ Multimodal fusion (CLIP)
- ✅ Transfer learning from large pretrained models

---

## 🔬 RESEARCH FINDINGS

### 1. BERT for Social Media Engagement (2024-2025)

#### Study 1: BERT-Based Emotion Framework (2025)
**Source:** JAIB 2025 - "Boosting Viewer Experience with Emotion-Driven Video Analysis"

**Key Findings:**
- **Model:** Pre-trained BERT with 110M parameters
- **Performance:** 83% accuracy, F1=0.83
- **Superiority:** Outperformed CNN (74%), LSTM (73%), SVM (72%)
- **Application:** Emotion recognition for video content engagement

**Relevance to Our Project:**
- Emotion-driven content = higher engagement
- BERT captures emotional context better than traditional ML
- Pretrained embeddings transfer well to social media domain

---

### 2. Indonesian BERT Models (IndoBERT) - 2024 Research

#### Study 2: Aspect-Based Sentiment Analysis (2024)
**Source:** MDPI Applied Sciences 2024, Universitas Diponegoro

**Performance:**
- **Aspect extraction accuracy:** 97.3%
- **Aspect F1-score:** 95.2%
- **Sentiment classification accuracy:** 97.9%
- **Sentiment F1-score:** 97.4%

**Model:** IndoBERT with fine-tuning on Indonesian student feedback

**Key Insight:** IndoBERT achieves near-perfect performance on Indonesian text!

---

#### Study 3: Hybrid IndoBERT + BiLSTM (2024)
**Source:** Wiley - Applied Computational Intelligence 2024

**Architecture:**
- IndoBERT base (last 4 hidden layers)
- + BiLSTM/BiGRU
- + Attention mechanism

**Performance:** Improved accuracy on Indonesian NLU benchmark

**Key Insight:** Combining IndoBERT embeddings with RNN captures both context and sequence

---

#### Study 4: Political Sentiment Analysis (2024)
**Source:** Indo-JC 2024

**Comparison:**
- **IndoBERT:** 70% accuracy
- **RoBERTa Indonesia:** 67% accuracy

**Dataset:** Indonesian General Sentiment Analysis Dataset

**Key Insight:** IndoBERT outperforms other Indonesian models

---

#### Study 5: Domain-Specific Fine-Tuning (2024)
**Source:** JISEBI 2024 - Indonesian Travel UGC

**Best Configuration:**
- Freeze last 6 layers
- Fine-tune top 6 layers
- **Validation loss:** 0.324
- **Aspect detection precision:** 0.85-0.89
- **Sentiment accuracy:** 0.84

**Key Insight:** Partial fine-tuning works better than full fine-tuning for small datasets!

---

### 3. Multimodal Transformers (Text + Image) - 2024 Research

#### Study 6: Vision-Language Models for Social Media (2024)
**Source:** arXiv 2024 - "Revisiting Vision-Language Features Adaptation"

**Model:** CLIP-adapter (pretrained CLIP + learnable layers)

**Key Findings:**
- ✅ Semantic inconsistency between image and text **increases** with popularity!
- ✅ CLIP-adapter improves representation for social media tasks
- ✅ Fine-tuning small adapter layers >> fine-tuning entire backbone

**Architecture:**
```
CLIP backbone (frozen)
    ↓
Adapter layers (learnable)
    ↓
Task-specific head (popularity prediction)
```

**Relevance:** Instagram posts often have image-text mismatch - CLIP can model this!

---

#### Study 7: Multi-Pop Model (2024)
**Source:** Wiley Expert Systems 2024

**Model:** Multimodal content-based popularity prediction

**Features:**
- Visual: CLIP image embeddings
- Text: BERT text embeddings
- Fusion: Concatenation + MLP

**Performance:**
- **Accuracy:** 82.0%
- **Superior to:** Single-modal methods

**Key Insight:** Multimodal features >> unimodal features

---

#### Study 8: Video Popularity Prediction (2024)
**Source:** arXiv 2024 - "MVP: SMP Challenge 2025 Video Track"

**Models Used:**
- **Visual:** CLIP, VideoMAE
- **Text:** BERT, RoBERTa
- **Ensemble:** Tree-based + hybrid models

**Best Practice:**
1. Extract features from pretrained backbones
2. Add structured context features
3. Use tree-based ensemble (XGBoost, LightGBM)

**Relevance:** Same approach applicable to Instagram Reels/videos!

---

### 4. Available Pretrained Models

#### Indonesian Language Models (HuggingFace)

| Model | Parameters | Training Data | Use Case |
|-------|-----------|---------------|----------|
| **indobenchmark/indobert-base-p1** | 110M | MLM + NSP | General Indonesian NLP |
| **indobenchmark/indobert-base-p2** | 110M | Improved dataset | Better performance |
| **indobenchmark/indobert-lite-base-p1** | 22M | Lightweight | Fast inference |
| **indobenchmark/indobert-large-p1** | 340M | Large scale | Best performance |
| **indolem/indobert-base-uncased** | 110M | 220M words | IndoLEM benchmark |
| **indolem/IndoBERTweet** | 110M | Indonesian Twitter | Social media specific! |

**Recommendation:** Use **indolem/IndoBERTweet** for Instagram (trained on social media!)

---

#### Multimodal Vision-Language Models

| Model | Size | Modality | Use Case |
|-------|------|----------|----------|
| **openai/clip-vit-base-patch32** | 151M | Image+Text | General multimodal |
| **openai/clip-vit-large-patch14** | 428M | Image+Text | Better performance |
| **Salesforce/blip-image-captioning-base** | 224M | Image→Text | Caption generation |
| **google/vit-base-patch16-224** | 86M | Image only | Visual features |
| **microsoft/git-base** | 681M | Image→Text | Instagram captions |

**Recommendation:** Use **CLIP** for image-text alignment features

---

#### Instagram-Specific Models

| Model | Base | Trained On | Download |
|-------|------|------------|----------|
| **mrSoul7766/git-base-instagram-cap** | Microsoft GIT | Instagram captions | HuggingFace |

**Note:** Fine-tuned specifically for Instagram-style captions!

---

## 🎯 PHASE 4 IMPLEMENTATION PLAN

### Approach 1: IndoBERT Text Embeddings (QUICK WIN)

**Goal:** Replace simple NLP features with IndoBERT embeddings

**Implementation:**
```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load IndoBERTweet (social media specific!)
tokenizer = AutoTokenizer.from_pretrained("indolem/IndoBERTweet")
model = AutoModel.from_pretrained("indolem/IndoBERTweet")

def extract_bert_features(caption):
    # Tokenize
    inputs = tokenizer(caption, return_tensors="pt",
                      truncation=True, max_length=128)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Use [CLS] token (sentence representation)
    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()

    # Or use mean pooling
    mean_embedding = outputs.last_hidden_state.mean(dim=1).numpy()

    return cls_embedding  # 768-dim vector
```

**Features Generated:**
- 768-dimensional caption embedding (IndoBERTweet)
- Captures: context, sentiment, semantic meaning

**Expected Improvement:** +0.05-0.10 R²

**Complexity:** ⭐⭐ (Medium - requires PyTorch)

---

### Approach 2: ViT Visual Embeddings (MEDIUM COMPLEXITY)

**Goal:** Replace OpenCV features with Vision Transformer embeddings

**Implementation:**
```python
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image

# Load Vision Transformer
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')

def extract_vit_features(image_path):
    # Load image
    image = Image.open(image_path)

    # Preprocess
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Get embeddings
    with torch.no_grad():
        outputs = vit_model(**inputs)

    # Use [CLS] token (image representation)
    image_embedding = outputs.last_hidden_state[:, 0, :].numpy()

    return image_embedding  # 768-dim vector
```

**Features Generated:**
- 768-dimensional image embedding (ViT)
- Captures: objects, composition, visual semantics

**Expected Improvement:** +0.08-0.15 R²

**Complexity:** ⭐⭐⭐ (Medium-High)

---

### Approach 3: CLIP Multimodal Alignment (ADVANCED)

**Goal:** Capture image-text relationship using CLIP

**Implementation:**
```python
from transformers import CLIPProcessor, CLIPModel

# Load CLIP
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

def extract_clip_features(image_path, caption):
    # Load image
    image = Image.open(image_path)

    # Process both modalities
    inputs = processor(text=caption, images=image,
                       return_tensors="pt", padding=True)

    # Get embeddings
    with torch.no_grad():
        outputs = clip_model(**inputs)

    # Image features
    image_embeds = outputs.image_embeds.numpy()  # 512-dim

    # Text features
    text_embeds = outputs.text_embeds.numpy()    # 512-dim

    # Similarity score (alignment)
    similarity = (image_embeds @ text_embeds.T).item()

    # Concatenate all features
    multimodal_features = {
        'clip_image_embedding': image_embeds,      # 512
        'clip_text_embedding': text_embeds,        # 512
        'clip_similarity': similarity,             # 1
        'clip_fusion': image_embeds * text_embeds  # 512 (element-wise)
    }

    return multimodal_features
```

**Features Generated:**
- 512-dim image embedding (CLIP)
- 512-dim text embedding (CLIP)
- 1 similarity score (alignment)
- 512-dim fusion features

**Total:** 1537 features from CLIP alone!

**Expected Improvement:** +0.15-0.25 R²

**Complexity:** ⭐⭐⭐⭐ (High)

---

### Approach 4: Hybrid Architecture (BEST PERFORMANCE)

**Goal:** Combine all transformer features + traditional features

**Architecture:**
```
Input Post
├── Image
│   ├── ViT embeddings (768)
│   ├── CLIP image embeddings (512)
│   └── OpenCV features (17)          → Total: 1297 visual features
│
├── Caption
│   ├── IndoBERTweet embeddings (768)
│   ├── CLIP text embeddings (512)
│   └── NLP features (14)             → Total: 1294 text features
│
└── Metadata
    ├── Temporal features (9)
    ├── Interaction features (5)
    └── CLIP alignment (1)             → Total: 15 meta features

TOTAL FEATURES: 2606

Feature Reduction (PCA/Autoencoder):
2606 → 128 dimensions

Model: HistGradientBoosting Ensemble
```

**Expected Performance:**
- **MAE:** 50-70 likes (target: <70) ✅
- **R²:** 0.40-0.55 (target: >0.35) ✅✅

**Complexity:** ⭐⭐⭐⭐⭐ (Very High)

**Requirements:**
- PyTorch
- Transformers library
- GPU (recommended for inference)
- 8-16GB RAM

---

## 📈 EXPECTED PERFORMANCE GAINS

### Feature Comparison

| Feature Type | Current | With Transformers | Improvement |
|--------------|---------|-------------------|-------------|
| **Text Features** | 14 (word lists) | 768 (IndoBERT) | **54x richer** |
| **Visual Features** | 17 (OpenCV) | 768 (ViT) | **45x richer** |
| **Multimodal** | 0 | 1537 (CLIP) | **∞ (new!)** |
| **Total Dims** | 28 | 2606 | **93x more** |

### Performance Projection

| Phase | Features | MAE | R² | Method |
|-------|----------|-----|-----|--------|
| Phase 2 (current) | 28 | 109.42 | 0.200 | NLP + Ensemble |
| **Phase 4a (IndoBERT)** | 768+28 | **~95** | **~0.25** | +Text embeddings |
| **Phase 4b (+ ViT)** | 1536+28 | **~75** | **~0.35** | +Visual embeddings |
| **Phase 4c (+ CLIP)** | 2606 | **~60** | **~0.45** | +Multimodal fusion |
| **Phase 4d (+ PCA+Ensemble)** | 128 reduced | **~50-70** | **~0.45-0.55** | Full pipeline |

**Target Achievement:**
- ✅ MAE < 70 (achieved in Phase 4c)
- ✅ R² > 0.35 (achieved in Phase 4b)

---

## 🚀 IMPLEMENTATION ROADMAP

### Week 1: IndoBERT Text Embeddings

**Day 1-2: Setup**
```bash
# Install dependencies
pip install torch transformers

# Download model (one-time)
python -c "from transformers import AutoModel; \
           AutoModel.from_pretrained('indolem/IndoBERTweet')"
```

**Day 3-4: Feature Extraction**
```python
# Create script: extract_bert_features.py
# Extract 768-dim embeddings for all 271 captions
# Save to: data/processed/bert_embeddings.csv
```

**Day 5: Training**
```python
# Update improve_model_v4.py
# Add BERT embeddings to feature set
# Train ensemble model
# Expected: MAE ~95, R² ~0.25
```

**Deliverable:** Phase 4a model with IndoBERT embeddings

---

### Week 2: Vision Transformer Features

**Day 1-2: ViT Setup**
```bash
# Download ViT model
python -c "from transformers import ViTModel; \
           ViTModel.from_pretrained('google/vit-base-patch16-224')"
```

**Day 3-5: Visual Feature Extraction**
```python
# Create script: extract_vit_features.py
# Extract 768-dim embeddings for all images
# Process 271 images (~5 min with GPU)
# Save to: data/processed/vit_embeddings.csv
```

**Day 6-7: Training**
```python
# Combine: BERT (768) + ViT (768) + baseline (28)
# Total: 1564 features
# Train with feature selection (keep top 200)
# Expected: MAE ~75, R² ~0.35
```

**Deliverable:** Phase 4b model with text + visual transformers

---

### Week 3: CLIP Multimodal Fusion

**Day 1-3: CLIP Feature Extraction**
```python
# Extract CLIP features for all posts
# Image embeds (512) + Text embeds (512) + Similarity (1)
# Save to: data/processed/clip_features.csv
```

**Day 4-5: Dimensionality Reduction**
```python
from sklearn.decomposition import PCA

# Reduce 2606 features → 128 features
# Preserve 95% variance
# Train on reduced features
```

**Day 6-7: Final Ensemble**
```python
# Ensemble: HistGB + XGBoost + RF
# Hyperparameter tuning with GridSearch
# Expected: MAE ~60, R² ~0.45
```

**Deliverable:** Phase 4c full transformer model

---

### Week 4: Optimization & Deployment

**Tasks:**
1. Model compression (quantization)
2. Inference optimization (batch processing)
3. Create prediction API
4. Documentation & paper writing

---

## ⚠️ CHALLENGES & SOLUTIONS

### Challenge 1: Computational Cost

**Problem:** Transformers require GPU, large RAM

**Solutions:**
- ✅ Use lightweight models (IndoBERT-lite, ViT-small)
- ✅ Batch processing during extraction
- ✅ Cache embeddings (extract once, use many times)
- ✅ Use Google Colab (free GPU) for extraction
- ✅ Quantize models (FP16 instead of FP32)

---

### Challenge 2: Model Size

**Problem:** Pretrained models are large (500MB - 2GB each)

**Solutions:**
- ✅ Download only once, cache locally
- ✅ Use `--cache-dir` parameter
- ✅ Share models across experiments
- ✅ Consider distilled models (smaller, faster)

---

### Challenge 3: Feature Dimensionality

**Problem:** 2606 features may cause overfitting

**Solutions:**
- ✅ **PCA:** Reduce to 128 dimensions (preserve 95% variance)
- ✅ **Feature Selection:** Keep top 200 features by importance
- ✅ **Autoencoder:** Learn compressed representation
- ✅ **L1/L2 Regularization:** Prevent overfitting

---

### Challenge 4: Indonesian Language Specificity

**Problem:** Some models not trained on Indonesian

**Solutions:**
- ✅ **Text:** Use IndoBERTweet (trained on Indonesian social media!)
- ✅ **Visual:** ViT/CLIP are language-agnostic (work globally)
- ✅ **Fine-tuning:** Fine-tune on our 271 posts (optional)

---

## 💰 COST-BENEFIT ANALYSIS

### Option 1: Simple IndoBERT (Week 1 only)

**Cost:**
- Time: 5 days
- Compute: CPU sufficient
- Complexity: Medium

**Benefit:**
- +0.05-0.10 R² improvement
- 768 rich text features
- Better sentiment understanding

**ROI:** ⭐⭐⭐⭐ (High - quick win!)

---

### Option 2: IndoBERT + ViT (Week 1-2)

**Cost:**
- Time: 10-12 days
- Compute: GPU recommended
- Complexity: Medium-High

**Benefit:**
- +0.13-0.20 R² improvement
- Text + Visual deep features
- Likely achieve R² > 0.35 target

**ROI:** ⭐⭐⭐⭐⭐ (Excellent - target achieved!)

---

### Option 3: Full CLIP Multimodal (Week 1-3)

**Cost:**
- Time: 15-20 days
- Compute: GPU required
- Complexity: High

**Benefit:**
- +0.20-0.30 R² improvement
- State-of-the-art features
- Image-text alignment captured
- Likely achieve MAE < 70 target

**ROI:** ⭐⭐⭐⭐⭐ (Excellent - both targets achieved!)

---

## 📚 IMPLEMENTATION EXAMPLE

### Minimal Working Example (IndoBERT only)

```python
#!/usr/bin/env python3
"""
Phase 4a: IndoBERT Text Embeddings
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load IndoBERTweet
print("Loading IndoBERTweet...")
tokenizer = AutoTokenizer.from_pretrained("indolem/IndoBERTweet")
model = AutoModel.from_pretrained("indolem/IndoBERTweet")

def extract_bert_embedding(caption):
    """Extract 768-dim BERT embedding from caption."""
    inputs = tokenizer(caption, return_tensors="pt",
                      truncation=True, max_length=128,
                      padding='max_length')

    with torch.no_grad():
        outputs = model(**inputs)

    # Use [CLS] token embedding
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embedding

# Load data
print("Loading data...")
df = pd.read_csv('fst_unja_from_gallery_dl.csv')

# Extract BERT embeddings
print("Extracting BERT embeddings for 271 captions...")
bert_embeddings = []

for idx, caption in enumerate(df['caption'].fillna('')):
    if (idx + 1) % 50 == 0:
        print(f"  Processed {idx + 1}/271...")

    embedding = extract_bert_embedding(str(caption))
    bert_embeddings.append(embedding)

# Convert to DataFrame
bert_features = pd.DataFrame(
    bert_embeddings,
    columns=[f'bert_dim_{i}' for i in range(768)]
)

# Load baseline features (from Phase 2)
baseline_df = pd.read_csv('data/processed/baseline_dataset.csv')

# Combine features
X_combined = pd.concat([
    baseline_df.drop(columns=['likes']),  # Baseline features
    bert_features                          # BERT embeddings
], axis=1)

y = baseline_df['likes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, np.log1p(y), test_size=0.3, random_state=42
)

# Train model
print("\nTraining HistGradientBoosting with BERT features...")
model = HistGradientBoostingRegressor(
    max_iter=300,
    max_depth=12,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred_test = np.expm1(model.predict(X_test))
y_test_orig = np.expm1(y_test)

mae = mean_absolute_error(y_test_orig, y_pred_test)
r2 = r2_score(y_test_orig, y_pred_test)

print(f"\n{'='*60}")
print("PHASE 4A RESULTS - IndoBERT Embeddings")
print(f"{'='*60}")
print(f"MAE:  {mae:.2f} likes")
print(f"R²:   {r2:.4f}")
print(f"Features: {X_combined.shape[1]} (28 baseline + 768 BERT)")
print(f"{'='*60}\n")

# Expected output:
# MAE:  ~90-95 likes (vs 109 current)
# R²:   ~0.23-0.27 (vs 0.20 current)
```

---

## 🎯 SUCCESS CRITERIA

### Phase 4a (IndoBERT) - Week 1

**Target:**
- ✅ MAE < 100 likes
- ✅ R² > 0.23
- ✅ +0.05 R² improvement

**Deliverable:**
- IndoBERT embeddings extracted
- Model trained & evaluated
- Performance documented

---

### Phase 4b (IndoBERT + ViT) - Week 2

**Target:**
- ✅ MAE < 80 likes
- ✅ R² > 0.33
- ✅ +0.13 R² total improvement

**Deliverable:**
- ViT embeddings extracted
- Combined model trained
- Feature importance analysis

---

### Phase 4c (Full Transformer) - Week 3

**Target:**
- ✅ MAE < 70 likes (FINAL TARGET!)
- ✅ R² > 0.40 (EXCEED TARGET!)
- ✅ +0.20 R² total improvement

**Deliverable:**
- CLIP features extracted
- Full multimodal model
- State-of-the-art performance

---

## 📝 REFERENCES & CITATIONS

### Academic Papers (2024-2025)

1. **BERT for Social Media Emotion** (2025)
   - JAIB 2025: "Boosting Viewer Experience with Emotion-Driven Video Analysis"
   - 83% accuracy with 110M parameter BERT

2. **IndoBERT Aspect-Based Sentiment** (2024)
   - MDPI Applied Sciences 2024
   - 97.9% sentiment classification accuracy

3. **CLIP for Social Media Popularity** (2024)
   - arXiv 2024: "Revisiting Vision-Language Features Adaptation"
   - CLIP-adapter for popularity prediction

4. **Multi-Pop Multimodal Model** (2024)
   - Wiley Expert Systems 2024
   - 82% accuracy with CLIP + BERT

5. **IndoBERT Political Sentiment** (2024)
   - Indo-JC 2024
   - 70% accuracy on Indonesian sentiment

---

### Pretrained Models

1. **indolem/IndoBERTweet**
   - Indonesian Twitter BERT (EMNLP 2021)
   - 110M parameters, trained on social media

2. **indobenchmark/indobert-base-p1**
   - General Indonesian BERT
   - MLM + NSP objectives

3. **openai/clip-vit-base-patch32**
   - Multimodal vision-language model
   - 151M parameters

4. **google/vit-base-patch16-224**
   - Vision Transformer for images
   - 86M parameters

---

## 🎓 NEXT STEPS - ACTION PLAN

### Immediate (This Week)

**Day 1-2: Environment Setup**
```bash
# Create new environment for transformers
conda create -n insta_transformers python=3.10
conda activate insta_transformers

# Install dependencies
pip install torch transformers pandas scikit-learn

# Test installation
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
```

**Day 3-5: IndoBERT Feature Extraction**
```bash
# Create extraction script
python extract_bert_features.py

# Expected output: data/processed/bert_embeddings.csv
# Size: 271 rows × 768 columns
```

**Day 6-7: Phase 4a Training**
```bash
# Train model with BERT features
python improve_model_v4_bert.py

# Expected: MAE ~95, R² ~0.25
```

---

### Short-term (Week 2)

**Add ViT visual embeddings**
- Extract ViT features for all images
- Combine with IndoBERT features
- Target: R² > 0.35

---

### Medium-term (Week 3-4)

**Full multimodal pipeline**
- CLIP feature extraction
- Dimensionality reduction (PCA)
- Final ensemble model
- Target: MAE < 70, R² > 0.40

---

### Long-term (Month 2-3)

**Paper Publication**
- Title: "Multimodal Transformer-Based Instagram Engagement Prediction for Indonesian Academic Institutions"
- Target: SINTA 2 journal
- Novel contribution: First study using IndoBERT + CLIP for Indonesian Instagram

---

## ✅ FINAL RECOMMENDATIONS

### Best Approach for Our Project:

**Phase 4b: IndoBERT + ViT (Week 1-2)**

**Why?**
1. ✅ Balanced effort vs reward
2. ✅ Likely achieves R² > 0.35 target
3. ✅ Manageable complexity
4. ✅ No GPU absolutely required (can use CPU, just slower)
5. ✅ Clear improvement over Phase 2

**Expected Results:**
- **MAE:** 70-80 likes ✅ (close to target)
- **R²:** 0.33-0.40 ✅ (exceeds target!)

**Timeline:** 10-12 days

**Complexity:** Medium-High (manageable)

---

### Alternative: Phase 4a Only (Week 1)

**If time/resources limited:**

**IndoBERT text embeddings only**
- Still get +0.05-0.10 R² improvement
- Much simpler implementation
- Quick win for paper

**Trade-off:** Won't fully achieve MAE < 70 target, but significant improvement

---

## 🎯 CONCLUSION

**Transformer-based approaches offer:**
1. ✅ **Massive feature richness** (768-2606 dims vs 28 current)
2. ✅ **Contextual understanding** (vs word-based features)
3. ✅ **Multimodal fusion** (image-text alignment)
4. ✅ **Transfer learning** (leverage pretrained knowledge)
5. ✅ **State-of-the-art performance** (likely achieve targets!)

**Recommendation:**
- **START:** Phase 4a (IndoBERT) this week
- **EXPAND:** Add ViT if Phase 4a successful
- **OPTIONAL:** CLIP for publication excellence

**Expected Final Performance:**
- **MAE:** 60-80 likes (vs 109 current)
- **R²:** 0.35-0.50 (vs 0.20 current)
- **Publication:** SINTA 2 journal quality

---

**Status:** Research complete, ready for implementation ✅
**Next:** Setup PyTorch environment & extract IndoBERT features
**Timeline:** 2-3 weeks to achieve targets

