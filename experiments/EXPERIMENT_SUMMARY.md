# EXPERIMENT SUMMARY REPORT

**Total Experiments:** 10
**Date:** 2025-10-04 00:34:11

---

## BEST MODELS

### Best Test MAE: baseline_cyclic_lag

- **Test MAE:** 125.69 likes
- **Test R²:** 0.073
- **Features:** 18
- **Description:** Baseline + Cyclic + Lag

### Best Test R²: temporal_vit

- **Test R²:** 0.130
- **Test MAE:** 126.53 likes
- **Features:** 786
- **Description:** Temporal + ViT

---

## ALL EXPERIMENTS

| Rank | Model | Test MAE | Test R² | Features | Overfit Gap |
|------|-------|----------|---------|----------|-------------|
| 1 | baseline_cyclic_lag | 125.69 | 0.073 | 18 | 0.018 |
| 2 | temporal_vit | 126.53 | 0.130 | 786 | 0.042 |
| 3 | baseline_vit_pca50 | 146.78 | -0.124 | 774 | 0.257 |
| 4 | baseline_cyclic | 149.24 | -0.059 | 12 | 0.173 |
| 5 | baseline_only | 158.03 | -0.163 | 6 | 0.174 |
| 6 | full_model | 170.08 | -0.223 | 1554 | 0.765 |
| 7 | temporal_bert | 170.42 | -0.207 | 786 | 0.738 |
| 8 | baseline_bert_pca100 | 194.45 | -0.155 | 774 | 0.502 |
| 9 | baseline_bert_nopca | 209.56 | -0.734 | 774 | 1.120 |
| 10 | baseline_bert_pca50 | 217.43 | -0.850 | 774 | 1.248 |

