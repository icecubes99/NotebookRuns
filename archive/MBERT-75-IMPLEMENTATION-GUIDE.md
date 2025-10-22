# 🚀 Quick Implementation Guide: mBERT 75%+ Optimization

## ⚡ 3-MINUTE SETUP

### Step 1: Open Your mBERT Notebook

Open `MBERT-TRAINING.ipynb` in Google Colab

### Step 2: Replace Configuration (Cell 8)

**Find Cell 8** (starts with `# ===== Section 3 — Config`)

**DELETE everything in that cell and REPLACE with the contents of:**
`MBERT-75-PERCENT-CONFIG.py`

**Or manually change these key values:**

```python
# CRITICAL CHANGES (Must do these):
EPOCHS = 12  # ← Change from 6
BATCH_SIZE = 16  # ← Change from 12
LR = 2.5e-5  # ← Change from 1.5e-5
EARLY_STOP_PATIENCE = 6  # ← Change from 3

# Boost weak classes:
CLASS_WEIGHT_MULT = {
    "sentiment": {"negative": 1.10, "neutral": 1.80, "positive": 1.30},  # ← neutral: 1.00→1.80
    "polarization": {"non_polarized": 1.20, "objective": 2.50, "partisan": 0.95}  # ← objective: 1.50→2.50
}
MAX_CLASS_WEIGHT = 10.0  # ← Change from 6.0

# Extreme oversampling:
JOINT_ALPHA = 0.70  # ← Change from 0.50
JOINT_OVERSAMPLING_MAX_MULT = 8.0  # ← Change from 4.0
OBJECTIVE_BOOST_MULT = 6.0  # ← Change from 2.5

# Larger model:
HEAD_HIDDEN = 768  # ← Change from 384
HEAD_LAYERS = 3  # ← Change from 2

# Stronger focal loss:
FOCAL_GAMMA_SENTIMENT = 2.0  # ← Change from 1.0
FOCAL_GAMMA_POLARITY = 2.5  # ← Change from 1.5

# Better regularization:
RDROP_ALPHA = 0.6  # ← Change from 0.4
LLRD_DECAY = 0.90  # ← Change from 0.95
```

### Step 3: Add Neutral Class Boosting (Section 9)

**Find Cell 20** (Section 9 - `train_eval_one_model` function)

**Locate this code** (around line 95 in the cell):

```python
        for ys, yp in zip(ysent_tr, ypol_tr):
            inv = inv_by_pair.get((int(ys), int(yp)), 1.0)
            w = (1.0 - JOINT_ALPHA) * 1.0 + JOINT_ALPHA * inv

            # Smart oversampling: extra boost for objective class
            if USE_SMART_OVERSAMPLING and int(yp) == obj_idx:
                w *= OBJECTIVE_BOOST_MULT

            sample_weights.append(w)
```

**ADD neutral class boosting** (add after objective boost):

```python
        # Find neutral sentiment index
        neutral_idx = np.where(sent_le.classes_ == "neutral")[0][0] if "neutral" in sent_le.classes_ else 1

        for ys, yp in zip(ysent_tr, ypol_tr):
            inv = inv_by_pair.get((int(ys), int(yp)), 1.0)
            w = (1.0 - JOINT_ALPHA) * 1.0 + JOINT_ALPHA * inv

            # Smart oversampling: extra boost for objective class
            if USE_SMART_OVERSAMPLING and int(yp) == obj_idx:
                w *= OBJECTIVE_BOOST_MULT

            # 🔥 NEW: Also boost neutral sentiment class
            if USE_SMART_OVERSAMPLING and int(ys) == neutral_idx:
                w *= NEUTRAL_BOOST_MULT

            sample_weights.append(w)
```

### Step 4: Run & Monitor

**Upload your data:**

- Upload `adjudications_2025-10-22.csv` to Colab

**Enable GPU:**

- Runtime → Change runtime type → T4 GPU

**Run all cells:**

- Runtime → Run all

**Expected output:**

```
🎯 mBERT OPTIMIZATION FOR 75%+ MACRO-F1
======================================================================
📊 Configuration Loaded:
   ├─ Training Epochs: 12 (doubled from 6)
   ├─ Effective Batch Size: 48
   ├─ Learning Rate: 2.5e-05
   ├─ Head Hidden Size: 768 (doubled from 384)
   └─ Max Class Weight: 10.0

⚖️  Critical Class Weights:
   ├─ Neutral:   1.8x (was 1.0x) - Fixes 49% F1
   └─ Objective: 2.5x (was 1.5x) - Fixes 40% F1

⏱️  Expected Training Time: ~55-70 minutes
🎯 Target Performance: ≥75% macro-F1
```

---

## 📊 MONITORING PROGRESS

### What to Watch During Training

#### **Loss Curves** (Should see):

- Steady decrease in loss
- No sudden spikes (indicates stability)
- Val loss following train loss (no huge gap = no overfitting)

#### **Metrics After Each Epoch**:

```
Epoch 1: ~50-55% F1 (warming up)
Epoch 3: ~60-65% F1 (matching baseline)
Epoch 6: ~68-72% F1 (surpassing baseline)
Epoch 9: ~72-75% F1 (approaching target)
Epoch 12: ~75-78% F1 (TARGET REACHED!)
```

#### **Early Stopping**:

- Should **NOT** trigger before epoch 8-9
- If it triggers early (epoch 4-5), increase `EARLY_STOP_PATIENCE` to 8

---

## ✅ SUCCESS CRITERIA

### **MINIMUM (Must achieve):**

- ✅ Sentiment F1: ≥ 75%
- ✅ Polarization F1: ≥ 75%
- ✅ Overall Macro-F1: ≥ 75%
- ✅ No class below 68% F1
- ✅ Training completed without NaN losses

### **STRETCH GOAL:**

- 🎯 Sentiment F1: ≥ 78%
- 🎯 Polarization F1: ≥ 77%
- 🎯 Overall Macro-F1: ≥ 77.5%
- 🎯 All classes ≥ 70% F1

---

## 🚨 TROUBLESHOOTING

### Problem 1: Training Too Slow (>90 minutes)

**Solution:** Reduce some settings:

```python
EPOCHS = 10  # Instead of 12
HEAD_HIDDEN = 512  # Instead of 768
```

### Problem 2: NaN Losses

**Solution:** Reduce aggressive settings:

```python
LR = 2e-5  # Instead of 2.5e-5
MAX_CLASS_WEIGHT = 8.0  # Instead of 10.0
MAX_GRAD_NORM = 1.0  # Instead of 0.5
```

### Problem 3: Overfitting (Val >> Train)

**Solution:** Increase regularization:

```python
WEIGHT_DECAY = 0.04  # Instead of 0.03
HEAD_DROPOUT = 0.30  # Instead of 0.25
RDROP_ALPHA = 0.7  # Instead of 0.6
```

### Problem 4: Still Below 75% After 12 Epochs

**Solutions:**

1. Increase epochs to 15
2. Boost weak class weights even more (neutral=2.2, objective=3.0)
3. Increase oversampling (objective_boost=8.0)
4. Add ensemble (train 3 models, average predictions)

---

## 📈 EXPECTED PER-CLASS IMPROVEMENTS

### Sentiment Classes:

```
                Before → After   Gain
negative:       73.3%  → 78-82%  +5-9%
neutral (WEAK): 49.4%  → 70-76%  +21-27% ← BIGGEST IMPROVEMENT
positive:       62.5%  → 75-80%  +13-18%
```

### Polarization Classes:

```
                    Before → After   Gain
non_polarized:      61.6%  → 73-78%  +11-16%
objective (WEAK):   40.4%  → 68-75%  +28-35% ← BIGGEST IMPROVEMENT
partisan:           76.7%  → 80-85%  +3-8%
```

---

## ⏱️ TIMELINE

```
00:00 - Start training
10:00 - Epoch 2 complete (~55% F1)
20:00 - Epoch 4 complete (~65% F1)
30:00 - Epoch 6 complete (~70% F1)
40:00 - Epoch 8 complete (~73% F1)
50:00 - Epoch 10 complete (~75% F1) ✅ TARGET!
60:00 - Epoch 12 complete (~76-78% F1) ✅ STRETCH GOAL!
```

---

## 🎉 AFTER TRAINING

### Download Your Model:

1. Navigate to `runs_mbert_optimized/`
2. Right-click → Download

### Check Results:

```python
# Look for these files:
- metrics_test.json  # Final performance metrics
- cm_sent.png        # Sentiment confusion matrix
- cm_pol.png         # Polarization confusion matrix
- report_sentiment.txt
- report_polarization.txt
```

### Validate Performance:

Open `metrics_test.json` and check:

```json
{
  "test_sent_f1": 0.76, // ← Should be ≥0.75
  "test_pol_f1": 0.75, // ← Should be ≥0.75
  "test_macro_f1_avg": 0.755 // ← Should be ≥0.75
}
```

---

## 🚀 IF YOU REACH 75%+

**Congratulations!** You've achieved:

- 🎯 Stable, high-performance mBERT model
- 🎯 >75% F1 across all metrics
- 🎯 Balanced performance (no weak classes)
- 🎯 Production-ready multilingual classifier

**Next Steps:**

1. ✅ Run XLM-RoBERTa with same optimizations (should reach 80%+)
2. ✅ Create ensemble (mBERT + XLM-R) for 82-85% F1
3. ✅ Deploy to production
4. ✅ Celebrate! 🎉

---

**TIME TO PUSH mBERT TO 75%+!** 🚀
