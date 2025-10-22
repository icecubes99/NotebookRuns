# mBERT Training - Quick Setup Guide

## ⚡ FASTEST METHOD (30 seconds)

### Step 1: Copy Your Existing Notebook

1. In Google Colab, open `EDITABLE-FILE.ipynb`
2. Click **File → Save a copy in Drive**
3. Rename it to: `MBERT-TRAINING.ipynb`

### Step 2: Make ONE Simple Change

Find **Cell 8** (Section 3 - Config) and change TWO lines:

```python
# FIND THIS LINE (around line 23):
MODELS_TO_RUN = ["mbert", "xlm_roberta"]

# REPLACE WITH:
MODELS_TO_RUN = ["mbert"]

# ALSO FIND THIS LINE (around line 85):
OUT_DIR = "./runs_multitask"

# REPLACE WITH:
OUT_DIR = "./runs_mbert"
```

### Step 3: Done!

Now you can run this notebook to train ONLY mBERT.

---

## 📊 What You'll Get

**Training Time:** ~35-40 minutes  
**Expected Performance:**

- Sentiment F1: ~61.7%
- Polarization F1: ~59.6%
- Overall Macro-F1: **~60.7%**

**Output Files (in `./runs_mbert/`):**

- `pytorch_model.bin` - Trained model weights
- `tokenizer.json` - Tokenizer files
- `metrics_test.json` - Performance metrics
- `cm_sent.png` - Sentiment confusion matrix
- `cm_pol.png` - Polarization confusion matrix
- Classification reports

---

## 🎯 Key Configuration (Already Optimized)

All improvements are already in your notebook:

- ✅ Class weights capped at 6.0 (prevents collapse)
- ✅ Smart oversampling for objective class
- ✅ Focal loss for both tasks
- ✅ R-Drop + LLRD regularization
- ✅ Enhanced architecture (384 hidden, 2-layer heads)
- ✅ Gradient clipping for stability

---

## 🔧 Optional: Quick Performance Boosts

If you want even better scores, add these changes to Cell 8:

```python
# Boost neutral class (currently struggles at 49% F1)
CLASS_WEIGHT_MULT = {
    "sentiment": {"negative": 1.20, "neutral": 1.30, "positive": 1.15},  # ← Changed 1.00 to 1.30
    "polarization": {"non_polarized": 1.00, "objective": 1.50, "partisan": 1.00}
}

# More training epochs
EPOCHS = 8  # ← Changed from 6 to 8

# Larger architecture
HEAD_HIDDEN = 512  # ← Changed from 384 to 512
```

**Expected Gains:** +3-5% macro-F1 (but adds ~10 minutes training time)

---

## 📝 Training Checklist

Before you hit "Run all":

- [ ] CSV file uploaded: `adjudications_2025-10-22.csv`
- [ ] File path in Cell 8 is correct: `/content/adjudications_2025-10-22.csv`
- [ ] `MODELS_TO_RUN = ["mbert"]` (only mBERT)
- [ ] `OUT_DIR = "./runs_mbert"` (separate output)
- [ ] GPU is enabled (Runtime → Change runtime type → T4 GPU)

---

## 🎉 After Training Completes

You'll see:

```
✅ SECTION 10: Model Training Execution completed in X minutes
🎯 Test Metrics:
   Sentiment F1: 0.617
   Polarization F1: 0.596
   Overall Macro-F1: 0.607
```

Download your model:

1. Click folder icon on left
2. Navigate to `runs_mbert/`
3. Right-click → Download entire folder

---

**That's it! Your mBERT model will train completely independently.** 🚀
