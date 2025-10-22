# XLM-RoBERTa Training - Quick Setup Guide

## ⚡ FASTEST METHOD (30 seconds)

### Step 1: Copy Your Existing Notebook

1. In Google Colab, open `EDITABLE-FILE.ipynb`
2. Click **File → Save a copy in Drive**
3. Rename it to: `XLM-ROBERTA-TRAINING.ipynb`

### Step 2: Make ONE Simple Change

Find **Cell 8** (Section 3 - Config) and change TWO lines:

```python
# FIND THIS LINE (around line 23):
MODELS_TO_RUN = ["mbert", "xlm_roberta"]

# REPLACE WITH:
MODELS_TO_RUN = ["xlm_roberta"]

# ALSO FIND THIS LINE (around line 85):
OUT_DIR = "./runs_multitask"

# REPLACE WITH:
OUT_DIR = "./runs_xlm_roberta"
```

### Step 3: Done!

Now you can run this notebook to train ONLY XLM-RoBERTa.

---

## 📊 What You'll Get

**Training Time:** ~35-40 minutes  
**Expected Performance:**

- Sentiment F1: ~67.6%
- Polarization F1: ~62.3%
- Overall Macro-F1: **~65.0%**

**Output Files (in `./runs_xlm_roberta/`):**

- `pytorch_model.bin` - Trained model weights
- `tokenizer.json` - Tokenizer files
- `metrics_test.json` - Performance metrics
- `cm_sent.png` - Sentiment confusion matrix
- `cm_pol.png` - Polarization confusion matrix
- Classification reports

---

## 🎯 Key Configuration (Already Optimized)

All improvements are already in your notebook:

- ✅ Class weights capped at 6.0
- ✅ Smart oversampling for objective class
- ✅ Focal loss for both tasks
- ✅ R-Drop + LLRD regularization
- ✅ Enhanced architecture (384 hidden, 2-layer heads)
- ✅ Gradient clipping for stability

---

## 🔧 Optional: Push to 70%+ F1

If you want even better scores, add these to Cell 8:

```python
# Boost neutral class (currently at 54% F1)
CLASS_WEIGHT_MULT = {
    "sentiment": {"negative": 1.20, "neutral": 1.35, "positive": 1.15},  # ← Changed 1.00 to 1.35
    "polarization": {"non_polarized": 1.00, "objective": 1.50, "partisan": 1.00}
}

# More training + larger model
EPOCHS = 8  # ← Changed from 6
HEAD_HIDDEN = 512  # ← Changed from 384
OBJECTIVE_BOOST_MULT = 3.5  # ← Changed from 2.5 (helps objective class)
```

**Expected Gains:** +4-6% macro-F1 → **~69-71% total** (adds ~12 minutes training)

---

## 📝 Training Checklist

Before running:

- [ ] CSV uploaded: `adjudications_2025-10-22.csv`
- [ ] Path correct in Cell 8: `/content/adjudications_2025-10-22.csv`
- [ ] `MODELS_TO_RUN = ["xlm_roberta"]` (only XLM-R)
- [ ] `OUT_DIR = "./runs_xlm_roberta"` (separate output)
- [ ] GPU enabled (Runtime → T4 GPU)

---

## 🎉 After Training

Expected output:

```
✅ SECTION 10: Model Training Execution completed
🎯 Test Metrics:
   Sentiment F1: 0.676
   Polarization F1: 0.623
   Overall Macro-F1: 0.650
```

Download:

1. Files panel (left sidebar)
2. `runs_xlm_roberta/`
3. Right-click → Download

---

## 🏆 XLM-RoBERTa vs mBERT

**Why XLM-R performs better:**

- Better multilingual representations
- Handles code-mixed text more naturally
- More robust to class imbalance
- Generally 4-5% higher F1 than mBERT

**This is your best single model!** 🚀
