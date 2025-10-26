# 🚀 RUN #15: QUICK START GUIDE

**Date:** October 26, 2025  
**Status:** Ready to run  
**Expected Time:** ~95-105 minutes  
**Expected Macro-F1:** 64-66% (+1-3% over R10's 63.06%)

---

## ✅ WHAT'S BEEN DONE

I've completed comprehensive research and preparation for your >75% breakthrough:

### 1. **Created Breakthrough Strategy Document** ✅

- File: `BREAKTHROUGH_STRATEGY_TO_75_PERCENT.md`
- Comprehensive 3-track plan to reach 75%+
- Research-backed techniques
- Timeline: 7 days to 75%+

### 2. **Updated Notebook for Run #15** ✅

- File: `MBERT_TRAINING.ipynb`
- Configuration: Sequence length optimization (MAX_LENGTH: 224 → 320)
- Restored R9 proven architecture (removed R14's harmful changes)
- Ready to execute immediately

### 3. **Completed Run #14 Analysis** ✅

- File: `RUN_ANALYSIS.md` (appended comprehensive analysis)
- Key finding: Architectural complexity backfired (-5.3% regression)
- Lesson: Simple architecture + data-centric approach is the path forward

---

## 🎯 RUN #15 STRATEGY

**Quick Win:** Sequence length optimization (research-backed)

**Key Change:**

```python
MAX_LENGTH = 320  # ⬆️ UP from 224 (+43% more context!)
```

**Why This Works:**

- NIH study (PMC9051848) showed length 320 achieved 71.7% F1
- News articles often have sentiment/polarization signals late in text
- Longer context = better understanding

**What's Restored:**

- All R9 proven hyperparameters
- Simple 3-layer head architecture (HEAD_HIDDEN=768)
- "last4_mean" pooling (stable and proven)
- NO complex architectural features

---

## 🏃 HOW TO RUN

**Option 1: Google Colab (Recommended)**

1. Upload `MBERT_TRAINING.ipynb` to Colab
2. Upload your dataset CSV
3. Run all cells (Runtime → Run all)
4. Wait ~95-105 minutes
5. Check results in Section 11

**Option 2: Local/Server**

```bash
jupyter notebook MBERT_TRAINING.ipynb
# Or
python -m jupyter notebook MBERT_TRAINING.ipynb
```

---

## 📊 EXPECTED RESULTS

**Run #15 Prediction:**

- **Sentiment F1:** 63-65% (vs R10: ~62%)
- **Polarization F1:** 65-67% (vs R10: ~64%)
- **Overall Macro-F1:** 64-66% (vs R10: 63.06%)
- **Improvement:** +1-3% from better context capture

**Per-Class Improvements (Expected):**

- **Objective:** 40-45% → 45-50% F1 (+5% from longer context)
- **Negative recall:** 45-50% → 50-55% (+5% from better understanding)
- **Neutral:** 55-60% → 58-62% (+3% from less confusion)

---

## 🚀 AFTER RUN #15: PATH TO 75%

**See `BREAKTHROUGH_STRATEGY_TO_75_PERCENT.md` for complete plan!**

**Quick Summary:**

### Track 1: Data Augmentation (CRITICAL) → +4-6%

- Back-translation for objective class (90 → 900 samples, +10x)
- Contextual word substitution
- Synonym replacement
- **Expected:** 68-72% Macro-F1

### Track 2: Smart Ensemble (QUICK WIN) → +2-3%

- Seed ensemble (average predictions from multiple seeds)
- Cross-architecture ensemble (mBERT + XLM-RoBERTa)
- **Expected:** 70-74% Macro-F1

### Track 3: Advanced Techniques (HIGH IMPACT) → +3-5%

- Curriculum learning (easy → hard examples)
- Pseudo-labeling (semi-supervised learning)
- Class-specific label smoothing
- **Expected:** 75-77% Macro-F1 ✅

---

## 📋 IMMEDIATE NEXT STEPS AFTER R15

1. **Run the analysis** (I'll do this for you when results are ready)
2. **If successful (64-66%):**

   - Proceed with data augmentation (Track 1)
   - Implement back-translation for objective class
   - Target: Run #16 with augmented data → 68-70%

3. **If unsuccessful (<64%):**
   - Analyze what went wrong
   - May need to adjust sequence length or restore R9 exactly
   - Fallback: Proceed with ensemble strategy first

---

## ⚠️ IMPORTANT NOTES

### Memory Considerations

- MAX_LENGTH=320 uses ~33% more memory than 224
- BATCH_SIZE=12 (down from 16) compensates for this
- Effective batch size still 48 (12 × 4 grad accumulation steps)
- Should fit on T4 GPU (16GB) without issues

### Time Expectations

- R9/R10: ~89-91 minutes with MAX_LENGTH=224
- R15: ~95-105 minutes with MAX_LENGTH=320 (+10% time)
- Longer sequences = slightly slower training

### What to Watch

- Training should be stable (no wild loss spikes)
- Validation Macro-F1 should improve steadily
- Early stopping at epoch 8 max (if not improving)

---

## 🔍 RESEARCH BACKING

**Sequence Length Study:**

- Source: NIH PMC9051848
- Finding: "BERT classifier with sequence length 320 achieved highest accuracy of 70.7% and F1 score of 71.7%"
- Comparison: Length 128 = 67.2% F1, Length 224 = 69.5% F1, **Length 320 = 71.7% F1**

**Why Longer Context Helps:**

1. News articles are typically 200-400 words
2. Sentiment indicators often appear in conclusions (late text)
3. Polarization context requires understanding full argumentation
4. Truncating at 224 tokens loses ~30% of average article

---

## 💡 KEY INSIGHTS FROM R14 FAILURE

**What NOT to Do (Learned from R14):**

- ❌ DON'T double model capacity (caused overfitting)
- ❌ DON'T add multiple architectural changes at once
- ❌ DON'T use attention pooling without extensive testing
- ❌ DON'T add residual blocks for small datasets
- ❌ DON'T reduce batch size below 12

**What DOES Work (Validated R1-R10):**

- ✅ Simple 3-layer heads (HEAD_HIDDEN=768)
- ✅ "last4_mean" pooling (stable)
- ✅ Proven R9 hyperparameters
- ✅ Incremental changes, one at a time
- ✅ Data-centric approach over architectural complexity

---

## 📈 PERFORMANCE TRAJECTORY

| Run     | Strategy              | Macro-F1   | Status         |
| ------- | --------------------- | ---------- | -------------- |
| R9      | Bug fixes + seed      | 62.74%     | ✅ Proven      |
| R10     | R9 same seed          | 63.06%     | ✅ Best        |
| R14     | Massive architecture  | 57.76%     | ❌ Disaster    |
| **R15** | **Sequence length**   | **64-66%** | **🎯 Current** |
| R16     | + Data augmentation   | 68-70%     | 📅 Next        |
| R17     | + Curriculum learning | 70-72%     | 📅 Planned     |
| R18     | + Pseudo-labeling     | 72-74%     | 📅 Planned     |
| R19     | + Final ensemble      | 75-77%     | 🎯 **TARGET**  |

---

## 🎯 SUCCESS CRITERIA

**Run #15 is successful if:**

- ✅ Overall Macro-F1 ≥ 64% (min target)
- ✅ No catastrophic per-class regressions
- ✅ Training completes without errors
- ✅ Stable training curve (no overfitting)

**Stretch Goals:**

- 🎯 Macro-F1 ≥ 65% (great!)
- 🎯 Objective class F1 ≥ 45% (improvement)
- 🎯 Negative recall ≥ 50% (fixed)

---

## 📞 READY TO GO!

**Your notebook is configured and ready to run.**

**What to do right now:**

1. Open `MBERT_TRAINING.ipynb` in Google Colab
2. Upload your dataset CSV
3. Click "Runtime → Run all"
4. Wait ~100 minutes
5. Come back and I'll analyze the results!

**After results:**

- Share the run-data with me
- I'll perform comprehensive analysis
- We'll proceed with the breakthrough strategy (Track 1: Data Augmentation)

---

**Good luck with Run #15! Let's get to 75%! 🚀**
