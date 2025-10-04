# Codebase Refactor Plan

**Date:** October 4, 2025
**Current Status:** Phase 4b Complete (271 posts → 396 posts)

---

## 📊 FILE CATEGORIZATION

### ✅ KEEP - Core Files (Actively Used)

**Python Scripts - Training & Extraction:**
- `extract_from_gallery_dl.py` - Extract metadata dari gallery-dl JSON
- `extract_bert_features.py` - Extract IndoBERT embeddings (Phase 4a)
- `extract_vit_features.py` - Extract ViT visual embeddings (Phase 4b)
- `improve_model_v4_bert.py` - Train Phase 4a model (BEST MAE: 98.94)
- `improve_model_v4_full.py` - Train Phase 4b model (BEST R²: 0.234)
- `predict.py` - CLI prediction tool
- `check_setup.py` - Environment validator

**Python Scripts - Analysis:**
- `compare_all_phases.py` - Compare Phase 0-4b performance
- `diagnostic_analysis.py` - Performance diagnostics

**Documentation - Essential:**
- `CLAUDE.md` - **MOST IMPORTANT** - Project guide for Claude Code
- `README.md` - Project overview
- `INSTAGRAM_DOWNLOAD_FIX.md` - Gallery-dl setup guide (just created)
- `PHASE4A_RESULTS.md` - IndoBERT results
- `PHASE4B_RESULTS.md` - Multimodal results (latest)
- `TRANSFORMER_RESEARCH.md` - Literature review (60+ pages)
- `requirements.txt` - Python dependencies

**Configuration:**
- `config.json` - Gallery-dl config
- `config.yaml` - ML pipeline config
- `cookies.txt` - Instagram authentication (DO NOT DELETE!)

**API & Deployment:**
- `api/main.py` - FastAPI endpoint (new)
- `app/streamlit_app.py` - Streamlit dashboard (new)

**Source Code (src/):**
- `src/features/` - Feature extraction modules
- `src/models/` - Model wrappers
- `src/utils/` - Config & logging utilities

---

### 📦 ARCHIVE - Old But Keep for Reference

**Python Scripts - Old Training (Phase 0-2):**
- `run_pipeline.py` - Phase 0-2 baseline training
- `improve_model.py` - Phase 1 (log transform)
- `improve_model_v2.py` - Phase 2 (NLP + ensemble)
- `improve_model_v3.py` - Phase 3 (experimental, possibly failed)

**Reason to keep:** Historical reference, reproduce earlier phases if needed

**Action:** Move to `archive/old_training/`

**Documentation - Historical:**
- `TRAINING_RESULTS.md` - Phase 1 results
- `RESEARCH_FINDINGS.md` - Phase 2 research
- `PHASE2_RESULTS.md` - Phase 2 comprehensive results
- `DEPLOYMENT_GUIDE.md` - Old deployment guide (before Phase 4)
- `FINAL_SUMMARY.md` - Pre-Phase 4 summary

**Reason to keep:** Historical context, publication reference

**Action:** Move to `archive/docs/`

---

### 🗑️ DELETE - Obsolete/Redundant Files

**Python Scripts - Obsolete:**
- `extract_visual_features.py` - OpenCV features (not used in Phase 4, replaced by ViT)
- `scripts/download_fresh_data.py` - Old download script (gallery-dl manual command better)
- `scripts/download_instagram.py` - Instaloader script (not used, gallery-dl chosen)
- `scripts/download_with_session.py` - Experimental download (not working)

**Reason:** Replaced by better methods, not used in current pipeline

**Phase 5 Experimental Scripts (Unfinished):**
- `phase5_1_advanced.py` - Incomplete experiment
- `phase5_ultraoptimize.py` - Incomplete experiment
- `PHASE5_FINAL_RESULTS.md` - Incomplete results

**Reason:** Phase 5 not complete, Phase 4b is current state

**Action:** DELETE or move to `experiments/incomplete/` if want to keep

**Documentation - Redundant:**
- `INSTAGRAM_DOWNLOAD_FIX_OLD.md` - Old backup (replaced by new version)
- `SESSION_SUMMARY.txt` - Text summary (redundant with MD files)
- `ULTRATHINK_SESSION_SUMMARY.md` - Session notes (redundant)
- `ULTRATHINK_FINAL_SUMMARY.md` - Session notes (redundant)
- `QUICKSTART.md` - Outdated quickstart (CLAUDE.md better)
- `ROADMAP.md` - Old roadmap (Phase 4b complete)
- `ROADMAP_SIMPLIFIED.md` - Old roadmap (Phase 4b complete)
- `README_IMPLEMENTATION.md` - Redundant with CLAUDE.md
- `IMPLEMENTATION_SUMMARY.md` - Redundant summary

**Reason:** Duplicate info, outdated, or temporary session notes

**Empty Folders:**
- `config/` - Empty
- `notebooks/` - Empty
- `tests/` - Empty

**Reason:** No content, not used

---

## 🏗️ PROPOSED NEW STRUCTURE

```
engagement_prediction/
├── README.md                          # Project overview
├── CLAUDE.md                          # ⭐ Main guide
├── requirements.txt                   # Dependencies
│
├── config.json                        # Gallery-dl config
├── config.yaml                        # ML pipeline config
├── cookies.txt                        # Instagram auth (gitignore!)
│
├── data/
│   ├── raw/                          # Original downloads
│   └── processed/                    # Processed features
│
├── models/                           # Trained models (.pkl)
│
├── src/                              # Source code
│   ├── features/                     # Feature extraction
│   ├── models/                       # Model wrappers
│   ├── training/                     # Training utilities
│   └── utils/                        # Config & logging
│
├── scripts/                          # Utility scripts
│   ├── extract_from_gallery_dl.py   # Metadata extraction
│   ├── extract_bert_features.py     # BERT extraction
│   ├── extract_vit_features.py      # ViT extraction
│   ├── train_phase4a.py             # Phase 4a training (renamed)
│   ├── train_phase4b.py             # Phase 4b training (renamed)
│   ├── predict.py                   # Prediction tool
│   ├── compare_phases.py            # Comparison (renamed)
│   └── check_setup.py               # Setup validator
│
├── api/
│   └── main.py                      # FastAPI endpoint
│
├── app/
│   └── streamlit_app.py             # Streamlit dashboard
│
├── docs/
│   ├── INSTAGRAM_DOWNLOAD_FIX.md    # Gallery-dl guide
│   ├── PHASE4A_RESULTS.md           # Current results
│   ├── PHASE4B_RESULTS.md           # Current results
│   ├── TRANSFORMER_RESEARCH.md      # Literature review
│   └── figures/                     # Visualizations
│
├── archive/
│   ├── old_training/                # Phase 0-3 scripts
│   │   ├── run_pipeline.py
│   │   ├── improve_model.py
│   │   ├── improve_model_v2.py
│   │   └── improve_model_v3.py
│   │
│   └── docs/                        # Historical docs
│       ├── TRAINING_RESULTS.md
│       ├── RESEARCH_FINDINGS.md
│       ├── PHASE2_RESULTS.md
│       ├── DEPLOYMENT_GUIDE.md
│       └── FINAL_SUMMARY.md
│
└── experiments/                     # Experimental work
    └── incomplete/                  # Unfinished experiments
        ├── phase5_1_advanced.py
        └── phase5_ultraoptimize.py
```

---

## ✏️ PROPOSED FILE RENAMES (for clarity)

**Before → After:**
- `improve_model_v4_bert.py` → `scripts/train_phase4a.py`
- `improve_model_v4_full.py` → `scripts/train_phase4b.py`
- `compare_all_phases.py` → `scripts/compare_phases.py`
- `diagnostic_analysis.py` → `scripts/diagnostic_analysis.py`

**Reason:** Clearer naming, organized in scripts/ folder

---

## 🎯 REFACTOR ACTIONS

### Step 1: Create New Folders
```bash
mkdir -p archive/old_training
mkdir -p archive/docs
mkdir -p experiments/incomplete
```

### Step 2: Move Archive Files
```bash
# Old training scripts
mv run_pipeline.py improve_model.py improve_model_v2.py improve_model_v3.py archive/old_training/

# Historical docs
mv TRAINING_RESULTS.md RESEARCH_FINDINGS.md PHASE2_RESULTS.md archive/docs/
mv DEPLOYMENT_GUIDE.md FINAL_SUMMARY.md archive/docs/
```

### Step 3: Move Experimental Files
```bash
mv phase5_1_advanced.py phase5_ultraoptimize.py experiments/incomplete/
mv PHASE5_FINAL_RESULTS.md experiments/incomplete/
```

### Step 4: Delete Obsolete Files
```bash
# Obsolete scripts
rm extract_visual_features.py
rm scripts/download_fresh_data.py scripts/download_instagram.py scripts/download_with_session.py

# Redundant docs
rm INSTAGRAM_DOWNLOAD_FIX_OLD.md SESSION_SUMMARY.txt
rm ULTRATHINK_SESSION_SUMMARY.md ULTRATHINK_FINAL_SUMMARY.md
rm QUICKSTART.md ROADMAP.md ROADMAP_SIMPLIFIED.md
rm README_IMPLEMENTATION.md IMPLEMENTATION_SUMMARY.md

# Empty folders
rmdir config/ notebooks/ tests/
```

### Step 5: Reorganize scripts/ Folder
```bash
# Move current training scripts to scripts/
mv extract_from_gallery_dl.py scripts/
mv extract_bert_features.py scripts/
mv extract_vit_features.py scripts/
mv predict.py scripts/
mv check_setup.py scripts/

# Rename for clarity
mv improve_model_v4_bert.py scripts/train_phase4a.py
mv improve_model_v4_full.py scripts/train_phase4b.py
mv compare_all_phases.py scripts/compare_phases.py
mv diagnostic_analysis.py scripts/
```

### Step 6: Update Import Paths
Update all scripts that import from moved files.

### Step 7: Update Documentation
- Update CLAUDE.md with new structure
- Update README.md with new file locations
- Add REFACTOR_CHANGELOG.md

---

## ⚠️ SAFETY CHECKS

**Before deleting anything:**
1. ✅ Verify git status (files tracked)
2. ✅ Create full backup: `tar -czf backup_$(date +%Y%m%d).tar.gz .`
3. ✅ Test that Phase 4a/4b scripts still work after moves
4. ✅ Verify no broken imports

**Files to NEVER delete:**
- `cookies.txt` (auth)
- `config.json` / `config.yaml` (configs)
- `requirements.txt` (dependencies)
- `CLAUDE.md` (main guide)
- Any `.pkl` models in `models/`
- Any data in `data/processed/` (embeddings)
- `src/` source code

---

## 📝 SUMMARY

**Total Files to Process:**
- **Keep in place:** 15 files
- **Move to archive:** 9 files
- **Move to experiments:** 3 files
- **Delete:** 12 files
- **Rename/reorganize:** 8 files

**Folders to remove:** 3 empty folders (config, notebooks, tests)

**Expected Result:**
- Cleaner root directory (only essential config files)
- Clear separation: scripts/, docs/, archive/, experiments/
- Better naming conventions (train_phase4a.py vs improve_model_v4_bert.py)
- Easier navigation for future work

---

## ✋ APPROVAL NEEDED

**User:** Review this plan and confirm before I proceed:
1. ✅ Approve all deletions?
2. ✅ Approve file moves?
3. ✅ Approve renames?
4. ✅ Any files to keep that I marked for delete/archive?

**Safety:** I will create a full backup before any operations.

---

**Generated:** October 4, 2025
**Status:** Awaiting approval
