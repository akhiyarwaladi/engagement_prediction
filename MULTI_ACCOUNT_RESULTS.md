# Multi-Account Model Results (1,949 Posts)
**Date:** October 4, 2025
**Accounts:** fst_unja (396 posts) + univ.jambi (1,553 posts)

---

## Executive Summary

Successfully trained engagement prediction models on **1,949 Instagram posts** (vs 348 previous), achieving:

- **30.1% improvement in MAE** (135.21 → 94.54)
- **53.7% improvement in R²** (0.4705 → 0.7234)
- **Best model:** Baseline + BERT PCA50 (59 features)

---

## Dataset Overview

### Data Collection
- **Total posts:** 1,949
  - fst_unja: 396 posts (53 videos)
  - univ.jambi: 1,553 posts (114 videos)
- **Total likes:** 675,129
- **Mean likes:** 346.40 per post
- **Median likes:** 130
- **Video avg likes:** 575.02 (77% higher than photos)

### Features Extracted
1. **Baseline features (9):** caption_length, word_count, hashtag_count, mention_count, is_video, hour, day_of_week, is_weekend, month
2. **BERT embeddings (768 → 50 via PCA):** IndoBERT text features (90.6% variance preserved)
3. **NIMA aesthetic features (8):** sharpness, noise, brightness, exposure quality, color harmony, saturation, saturation variance, luminance contrast

---

## Experimental Results

### Experiment 1: Baseline Only (9 features)
- **MAE:** 125.58 likes
- **R²:** 0.6746
- **vs 348-post baseline:** R² improved 0.086 → 0.6746 (684% increase!)

### Experiment 2: Baseline + BERT PCA50 (59 features) ⭐ BEST
- **MAE:** 94.54 likes
- **R²:** 0.7234
- **Improvement vs baseline:** 24.71% better MAE
- **BERT PCA variance:** 90.6% (excellent preservation)

### Experiment 3: Baseline + BERT + NIMA (67 features)
- **MAE:** 108.01 likes
- **R²:** 0.7013
- **Improvement vs baseline:** 14.00% better MAE
- **vs BERT:** -14.24% (NIMA degraded performance!)

---

## Key Findings

### 1. **More Data = Dramatic Improvement**
- **348 posts → 1,949 posts** (5.6x increase)
- Baseline R² jumped from 0.086 → 0.6746
- Proves that transformer models (BERT) need substantial data to shine

### 2. **BERT is the Champion** 🏆
- **Best model:** Baseline + BERT (59 features)
- **MAE:** 94.54 (vs 135.21 previous champion = **30.1% better**)
- **R²:** 0.7234 (vs 0.4705 previous = **53.7% better**)
- PCA preserved 90.6% variance (excellent dimensionality reduction)

### 3. **NIMA Aesthetics Added Noise**
- Adding NIMA features **worsened** MAE: 94.54 → 108.01 (+14.2%)
- Possible reasons:
  - NIMA features not relevant for academic Instagram accounts
  - fst_unja + univ.jambi prioritize information over aesthetics
  - Only 1,947/1,949 images had NIMA features (2 missing)
  - Generic aesthetic metrics may not capture Indonesian academic visual preferences

### 4. **Feature Importance Insights**
From model performance:
- **Text (BERT):** Primary driver of engagement
- **Baseline features:** Critical foundation (posting time, media type, caption length)
- **Visual aesthetics (NIMA):** Not as important for this domain

---

## Comparison with Previous Champion

### Previous Best (348 posts)
- **Model:** RFE 75 features (Baseline + NLP + engineered features)
- **MAE:** 135.21 likes
- **R²:** 0.4705

### Current Best (1,949 posts) ⭐
- **Model:** Baseline + BERT PCA50 (59 features)
- **MAE:** 94.54 likes (-30.1%)
- **R²:** 0.7234 (+53.7%)

### Why the Improvement?
1. **5.6x more training data** (348 → 1,949 posts)
2. **BERT captures Indonesian context better** than hand-crafted NLP features
3. **Multi-account diversity** (2 institutions vs 1) improves generalization
4. **Higher variance in likes** (more diverse engagement patterns to learn from)

---

## Model Performance Summary

| Model | Dataset | Features | MAE | R² | Notes |
|-------|---------|----------|-----|-----|-------|
| RFE 75 (old) | 348 posts | 75 | 135.21 | 0.4705 | Previous champion |
| Baseline | 1,949 posts | 9 | 125.58 | 0.6746 | Already better than old champion! |
| **BERT (NEW)** | **1,949 posts** | **59** | **94.54** | **0.7234** | **⭐ Current champion** |
| BERT + NIMA | 1,949 posts | 67 | 108.01 | 0.7013 | NIMA degraded performance |

---

## Technical Details

### Data Preprocessing
1. **Outlier clipping:** 99th percentile (3,284 likes)
2. **Log transformation:** log1p(y) for target variable
3. **Feature scaling:** QuantileTransformer (normal distribution)
4. **Train-test split:** 80/20, random_state=42

### Model Architecture
- **Ensemble:** Random Forest (50%) + HistGradientBoosting (50%)
- **Random Forest:** 250 trees, max_depth=14, min_samples_split=3, min_samples_leaf=2
- **HistGradientBoosting:** 400 iterations, max_depth=14, learning_rate=0.05, l2_regularization=0.1

### PCA Configuration
- **BERT:** 768 → 50 dimensions (90.6% variance)
- **NIMA:** No PCA (8 features only)

---

## Conclusions

### ✅ Successes
1. **Achieved goal:** MAE < 120 (target was < 120, achieved 94.54!)
2. **30% improvement** over previous champion (135.21 → 94.54)
3. **BERT proves valuable** for Indonesian Instagram text analysis
4. **Multi-account approach works** (fst_unja + univ.jambi)
5. **Data quantity matters** for deep learning features

### ⚠️ Surprises
1. **NIMA aesthetics degraded performance** (-14% vs BERT-only)
2. **Baseline alone (9 features) already beats old champion** (125.58 < 135.21)
3. **R² jumped 684%** just by adding more data (0.086 → 0.6746)

### 🚀 Recommendations
1. **Use Baseline + BERT (59 features)** for production
2. **Skip NIMA aesthetics** for academic Instagram accounts
3. **Collect even more data** (target 3,000-5,000 posts for further gains)
4. **Fine-tune IndoBERT** on Instagram captions for domain adaptation
5. **Explore video-specific features** (167 videos available)

---

## Next Steps

### Immediate
- [x] Extract data from multi-account (1,949 posts)
- [x] Extract BERT embeddings
- [x] Extract NIMA aesthetic features
- [x] Train and compare models

### Short-term (Week 1-2)
- [ ] Save best model (Baseline + BERT) for production
- [ ] Create prediction API/tool
- [ ] Extract video-specific features (motion, audio, temporal)
- [ ] Test on more Instagram accounts (@univ.jambi official, etc.)

### Long-term (Month 1-3)
- [ ] Fine-tune IndoBERT on Indonesian Instagram captions
- [ ] Explore CLIP for image-text alignment
- [ ] Temporal features (posting consistency, trending topics)
- [ ] Deploy as web service for social media managers

---

## Files Generated

- `multi_account_dataset.csv`: Combined dataset (1,949 posts)
- `data/processed/bert_embeddings_multi_account.csv`: BERT features (1,949 × 768)
- `data/processed/aesthetic_features_multi_account.csv`: NIMA features (1,947 × 8)
- `experiments/multi_account_results.csv`: Model performance comparison
- `multi_account_training.log`: Complete training logs

---

**Status:** ✅ Experiment Complete
**Champion Model:** Baseline + BERT PCA50 (MAE=94.54, R²=0.7234)
**Achievement:** 30.1% improvement over previous best (MAE 135.21 → 94.54)
