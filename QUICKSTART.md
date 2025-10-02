# ⚡ QUICK START - 5 Minutes to First Results!

## Step 1: Install (30 seconds)

```bash
pip install -r requirements.txt
```

## Step 2: Check Setup (10 seconds)

```bash
python3 check_setup.py
```

**Expected:** All ✅ checks pass

## Step 3: Train Model (30 seconds)

```bash
python3 run_pipeline.py
```

**Expected Output:**
```
================================================================================
PIPELINE EXECUTION SUCCESSFUL
================================================================================

📊 Final Evaluation Metrics:
--------------------------------------------------------------------------------
  MAE_test            : 58.45
  R2_test             : 0.568
--------------------------------------------------------------------------------

✅ MAE Target: ACHIEVED (58.45 <= 70)
✅ R² Target: ACHIEVED (0.568 >= 0.50)

DONE! 🎉
```

## Step 4: Make Prediction (5 seconds)

```bash
python3 predict.py \
  --caption "Selamat wisuda mahasiswa FST UNJA! #wisuda #fstunja" \
  --hashtags 3 \
  --date "2025-10-15 10:30"
```

**Expected Output:**
```
📊 Prediction Results:
--------------------------------------------------------------------------------
  Predicted Likes: 287
  Engagement Rate: 6.20%
  Performance: 🟡 Medium
--------------------------------------------------------------------------------

💡 Recommendations:
  1. ✅ Good timing! This is typically a high-engagement hour
  2. 🎥 Videos often get 20-30% more engagement than photos
```

---

## 🎉 That's It!

You now have:
- ✅ Trained Random Forest model
- ✅ Performance evaluation (MAE, R²)
- ✅ Feature importance analysis
- ✅ Prediction capability

---

## 📊 Check Results

```bash
# View training log
cat logs/training.log

# View plots
ls docs/figures/
# - feature_importance.png
# - predictions.png
```

---

## 🚀 Next Steps

1. **Analyze Results:**
   - Check `docs/figures/` for visualizations
   - Review feature importance

2. **Test Different Scenarios:**
   ```bash
   # High engagement scenario
   python3 predict.py --caption "Mahasiswa FST juara kompetisi nasional!" --hashtags 5 --date "2025-10-15 10:00"

   # Low engagement scenario
   python3 predict.py --caption "Info" --hashtags 1 --date "2025-10-14 23:00" --video
   ```

3. **Read Full Documentation:**
   - `README_IMPLEMENTATION.md` - Complete guide
   - `IMPLEMENTATION_SUMMARY.md` - What was built

4. **(Optional) Run Streamlit App:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

---

## ❓ Troubleshooting

### Error: "Module not found"
```bash
pip install --upgrade -r requirements.txt
```

### Error: "Data file not found"
```bash
# Check if CSV exists
ls -la fst_unja_from_gallery_dl.csv
```

### Low Performance
```bash
# Check data quality
python3 -c "import pandas as pd; df = pd.read_csv('fst_unja_from_gallery_dl.csv'); print(df['likes'].describe())"
```

---

## 📝 Quick Reference

### Train Model:
```bash
python3 run_pipeline.py
```

### Predict:
```bash
python3 predict.py --caption "TEXT" --hashtags N --date "YYYY-MM-DD HH:MM"
```

### Check Status:
```bash
python3 check_setup.py
```

---

**Total Time: 5 minutes ⚡**

**Questions? Check `README_IMPLEMENTATION.md`**
