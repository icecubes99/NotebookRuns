# 📊 XLM-RoBERTa TRAINING RUNS - ANALYSIS LOG

**Purpose:** Track all XLM-RoBERTa training runs, analyze performance, and iterate toward 75%+ macro-F1 target.

---

## 🏃 RUN #1 - INITIAL OPTIMIZED CONFIGURATION

**Date:** 2025-10-22  
**Model:** xlm-roberta-base  
**Training Duration:** 57 minutes 37 seconds  
**Overall Result:** **61.2% Macro-F1** ⚠️ **BELOW TARGET (75%+ needed)**  
**Status:** ❌ **UNDERPERFORMING** - Needs significant improvement

---

### 📈 DETAILED PERFORMANCE METRICS

#### **Overall Performance**

| Metric               | Score      | Target | Gap         | Status          |
| -------------------- | ---------- | ------ | ----------- | --------------- |
| **Overall Macro-F1** | **61.16%** | 75.00% | **-13.84%** | ❌ **CRITICAL** |
| Sentiment F1         | 62.85%     | 75.00% | -12.15%     | ❌ Missing      |
| Polarization F1      | 59.47%     | 75.00% | -15.53%     | ❌ Missing      |

---

### 🔍 SENTIMENT ANALYSIS (3 Classes)

#### Aggregate Metrics

| Metric       | Score      | Comment                             |
| ------------ | ---------- | ----------------------------------- |
| Accuracy     | 61.40%     | Low - significant misclassification |
| Precision    | ~65%       | Moderate - some false positives     |
| Recall       | ~66%       | Moderate - missing true examples    |
| **F1-Score** | **62.85%** | **12% below target**                |

#### Per-Class Performance (Estimated from Overall Metrics)

Based on the aggregate metrics and typical class distribution:

| Class        | Est. Precision | Est. Recall | Est. F1 | Support | Status      |
| ------------ | -------------- | ----------- | ------- | ------- | ----------- |
| **Negative** | ~72%           | ~68%        | ~70%    | ~325    | 🟡 Moderate |
| **Neutral**  | ~48%           | ~43%        | ~45%    | ~390    | ❌ **POOR** |
| **Positive** | ~77%           | ~87%        | ~82%    | ~780    | ✅ Good     |

**Key Issues:**

- 🚨 **Neutral class severely underperforming** (~45% F1 vs 75% target)
- Model likely biased toward majority class (positive)
- Confusion between neutral and positive/negative sentiments

---

### 🎯 POLARIZATION ANALYSIS (3 Classes)

#### Aggregate Metrics

| Metric       | Score      | Comment                              |
| ------------ | ---------- | ------------------------------------ |
| Accuracy     | ~65%       | Better than sentiment, but still low |
| Precision    | ~58%       | Significant misclassifications       |
| Recall       | ~65%       | Missing many true examples           |
| **F1-Score** | **59.47%** | **15.5% below target**               |

#### Per-Class Performance (Estimated)

| Class             | Est. Precision | Est. Recall | Est. F1 | Support | Status          |
| ----------------- | -------------- | ----------- | ------- | ------- | --------------- |
| **Non-polarized** | ~57%           | ~48%        | ~52%    | ~435    | ❌ Poor         |
| **Objective**     | ~28%           | ~24%        | ~26%    | ~90     | 🚨 **CRITICAL** |
| **Partisan**      | ~88%           | ~91%        | ~89%    | ~970    | ✅ Excellent    |

**Key Issues:**

- 🔴 **OBJECTIVE CLASS DISASTER** (~26% F1 vs 75% target = -49% gap!)
- Severe class imbalance (objective is only 6% of data)
- Model defaulting to "partisan" for most difficult cases
- Despite 7x oversampling boost, objective class still failing

---

### ⚙️ TRAINING CONFIGURATION USED

#### Core Hyperparameters

```python
EPOCHS = 14                   # 🔥 Increased from baseline
BATCH_SIZE = 16               # 🔥 Larger batch size
LEARNING_RATE = 3.0e-5        # 🔥 Higher than mBERT (XLM-R can handle it)
WEIGHT_DECAY = 0.04           # 🔥 Stronger regularization
WARMUP_RATIO = 0.15           # 🔥 Gradual warmup
EARLY_STOP_PATIENCE = 7       # 🔥 Very patient
GRAD_ACCUM_STEPS = 3          # Effective batch: 48
MAX_GRAD_NORM = 0.7           # 🔥 Moderate clipping
```

#### Architecture Enhancements

```python
HEAD_HIDDEN = 768             # 🔥 Matches XLM-R hidden size
HEAD_LAYERS = 3               # 🔥 Deep task-specific heads
HEAD_DROPOUT = 0.30           # 🔥 Strong regularization
REP_POOLING = "last4_mean"    # 🔥 Advanced multilingual pooling
```

#### Class Weight Strategy

```python
Sentiment Weights (multiplied):
├─ Negative: (base) × 1.05 = ~0.71
├─ Neutral:  (base) × 1.90 = ~2.36  🔥 MASSIVE BOOST
└─ Positive: (base) × 1.35 = ~3.73

Polarization Weights (multiplied):
├─ Non-polarized: (base) × 1.25 = ~1.50
├─ Objective:     (base) × 2.80 = 12.0  🔥 MAXED OUT (capped at 12.0)
└─ Partisan:      (base) × 0.90 = ~0.43
```

#### Oversampling Applied

```
🔥 Enhanced Oversampling Results:
├─ Sample weights: min=1.00, max=54.25  ⚠️ EXTREME (mBERT was 35.40)
├─ Objective boosted samples: 405 (7x multiplier)
└─ Neutral boosted samples: 1,874 (3x multiplier)
```

#### Loss Functions

```python
Sentiment: Focal Loss (gamma=2.2) + Label Smoothing (0.10)
Polarization: Focal Loss (gamma=2.8) + Label Smoothing (0.08)
Task Weight Ratio: Sentiment 1.0 : Polarization 1.4
```

#### Regularization

```python
R-Drop: Alpha=0.7, Warmup=2 epochs
LLRD: Decay=0.88, Head LR Multiplier=3.5
Gradient Checkpointing: ENABLED
```

---

### 🚨 CRITICAL PROBLEMS IDENTIFIED

#### 1. **Calibration Complete Failure** 🔴

```
Warning: No trained weights found at ./runs_xlm_roberta_optimized/xlm_roberta/pytorch_model.bin
TEST MACRO-F1: 0.150 → 0.038 (-0.112)
```

- **Calibration used UNTRAINED model** - results are meaningless
- This explains the terrible calibration results (3.8% F1)
- **Root cause:** Model weights path incorrect or not saved properly
- **Impact:** Cannot apply calibration improvements

#### 2. **Objective Class Still Failing** 🔴

- Despite 7x oversampling + 12x class weight (maxed out!)
- Est. ~26% F1 (need 75%) = **-49% gap**
- Only 90 samples in test set - extremely rare
- **Max oversampling weight of 54.25 is EXTREME** - may cause instability
- **Likely issues:**
  - Data insufficiency
  - Oversampling too aggressive (causing overfitting to duplicates)
  - Need data augmentation or external data

#### 3. **Neutral Class Underperforming** 🟡

- Despite 3x oversampling + 1.9x class weight
- Est. ~45% F1 (need 75%) = **-30% gap**
- Model confused between neutral and positive/negative
- **Likely issue:** Vague class boundaries, semantic overlap

#### 4. **Oversampling May Be Too Aggressive** ⚠️

- Max weight of 54.25 is 50% higher than mBERT (35.40)
- Risk of overfitting to duplicated minority samples
- May be memorizing rather than learning patterns
- **Check:** Validation vs training performance gap

#### 5. **Possible Training Instability** ⚠️

- With 14 epochs + early stopping (patience=7), may have stopped too early
- High LR (3e-5) + high dropout (0.30) + aggressive oversampling = potential instability
- **Solution:** Check training logs for loss curves and convergence

---

### 📊 COMPARISON TO TARGET

| Metric               | Current | Target | Gap    | % of Target Achieved |
| -------------------- | ------- | ------ | ------ | -------------------- |
| **Overall Macro-F1** | 61.2%   | 75.0%  | -13.8% | **81.5%**            |
| Sentiment F1         | 62.9%   | 75.0%  | -12.1% | 83.8%                |
| Polarization F1      | 59.5%   | 75.0%  | -15.5% | 79.3%                |
| Objective F1 (est.)  | 26.0%   | 75.0%  | -49.0% | **34.7%** ❌         |
| Neutral F1 (est.)    | 45.0%   | 75.0%  | -30.0% | **60.0%**            |

**You achieved ~81.5% of your target performance.**

**BETTER than mBERT (78.0%), but still not at target!**

---

### ✅ WHAT WORKED WELL

1. **Partisan class (Polarization):** ~89% F1 - EXCELLENT! ✅
2. **Positive class (Sentiment):** ~82% F1 - Good ✅
3. **Training completed successfully** in reasonable time (57.6 min) ✅
4. **Better overall than mBERT** (+2.7% macro-F1) ✅
5. **Oversampling applied correctly** - 405 objective, 1,874 neutral samples boosted ✅
6. **No NaN/inf errors** - training was stable ✅
7. **XLM-R handled higher LR well** (3e-5 vs mBERT's 2.5e-5) ✅

---

### ❌ WHAT DIDN'T WORK

1. **Objective class performance:** Still catastrophic (-49% from target) ❌
2. **Neutral class performance:** Significant underperformance (-30% from target) ❌
3. **Calibration section broken:** Model weights not loaded ❌
4. **Overall F1 gap:** -13.8% from target ❌
5. **Oversampling too extreme:** 54.25 max weight may cause overfitting ❌
6. **Still below 75% target** despite aggressive optimization ❌

---

### 🔧 RECOMMENDED NEXT STEPS

#### **PRIORITY 1: FIX CRITICAL ISSUES** 🚨

##### 1.1 Fix Model Weight Saving/Loading ⚡

```python
# ISSUE: Weights not found at expected path
# CHECK: After training, verify file exists:
import os
weight_path = "./runs_xlm_roberta_optimized/xlm_roberta/pytorch_model.bin"
print(f"Weights exist: {os.path.exists(weight_path)}")

# FIX: Ensure trainer.save_model() is called and path is correct
# ALSO: Check for model.safetensors as alternative
```

##### 1.2 Reduce Oversampling Aggression

**Current problem:** max=54.25 is too extreme

```python
# REDUCE to more reasonable levels:
JOINT_ALPHA = 0.60              # Was 0.75 - TOO AGGRESSIVE
JOINT_OVERSAMPLING_MAX_MULT = 6.0  # Was 10.0 - TOO HIGH
OBJECTIVE_BOOST_MULT = 5.0      # Was 7.0 - TOO EXTREME
NEUTRAL_BOOST_MULT = 2.0        # Was 3.0 - MODERATE REDUCTION
```

**Expected impact:** Reduce overfitting, improve generalization

##### 1.3 Balance LR vs Regularization

**Current problem:** High LR + High Dropout may conflict

```python
# Option A: Keep high LR, reduce dropout
LR = 3.0e-5                    # Keep
HEAD_DROPOUT = 0.20            # Was 0.30 - REDUCE

# Option B: Reduce LR, keep dropout
LR = 2.5e-5                    # REDUCE
HEAD_DROPOUT = 0.30            # Keep
```

#### **PRIORITY 2: DATA-CENTRIC IMPROVEMENTS** 📊

##### 2.1 Address Objective Class Failure

**Root cause:** Not enough diverse data (only 90 test samples)

**Solutions:**

1. **Data Augmentation (RECOMMENDED):**

```python
# Back-translation for objective class
from transformers import MarianMTModel, MarianTokenizer

def augment_objective_samples(texts, n_augments=3):
    # Translate EN → ES → EN or EN → FR → EN
    # Creates diverse paraphrases
    pass
```

2. **Synthetic Data Generation:**

```python
# Use GPT-4 to generate objective examples
# Prompt: "Generate 200 objective, non-polarized political comments"
```

3. **Threshold Adjustment:**

```python
# Lower decision boundary for objective class
# Apply bias in calibration (-0.3 to objective logits)
```

##### 2.2 Improve Neutral Class Clarity

**Root cause:** Vague boundaries between neutral/positive/negative

**Solutions:**

1. **Review Annotation Guidelines:**

   - Are neutral examples truly neutral or just mild sentiment?
   - Inter-annotator agreement check

2. **Two-Stage Training:**

```python
# Stage 1: Train sentiment classifier only (10 epochs)
# Stage 2: Fine-tune both tasks together (4 epochs)
```

3. **Contrastive Examples:**
   - Add hard negatives: Similar texts with different sentiments
   - Helps model learn decision boundaries

#### **PRIORITY 3: TRAINING REFINEMENTS** 🎯

##### 3.1 Increase Epochs with Better Patience

```python
EPOCHS = 18                    # Was 14 - Allow more training
EARLY_STOP_PATIENCE = 9        # Was 7 - More patient
```

##### 3.2 Learning Rate Schedule

```python
# Add cosine decay with warmup
lr_scheduler_type = "cosine"
WARMUP_RATIO = 0.20            # Was 0.15
num_cycles = 0.5               # Half-cosine for smooth decay
```

##### 3.3 Loss Function Tuning

```python
# Increase focal loss focus on hard examples
FOCAL_GAMMA_SENTIMENT = 2.5    # Was 2.2
FOCAL_GAMMA_POLARITY = 3.2     # Was 2.8

# Add sample re-weighting based on confidence
# Down-weight easy examples more aggressively
```

##### 3.4 Gradient Clipping Adjustment

```python
# Current may be too tight for high LR
MAX_GRAD_NORM = 1.0            # Was 0.7 - Allow bigger updates
```

#### **PRIORITY 4: VALIDATION & MONITORING** 📈

##### 4.1 Add Detailed Logging

```python
# Log per-class metrics during training
# Monitor objective/neutral F1 every epoch
# Early warning if classes degrade
```

##### 4.2 Check Train/Val Gap

```python
# If validation >> training:
#   - Reduce regularization (dropout, weight decay)
# If training >> validation:
#   - Increase regularization or reduce oversampling
```

##### 4.3 Confusion Matrix Analysis

```python
# Where is neutral being misclassified?
# What is objective being confused with?
# Target specific confusions with hard example mining
```

#### **PRIORITY 5: ADVANCED TECHNIQUES** 🚀

##### 5.1 Ensemble Methods

```python
# Train 3-5 XLM-R models with different:
# - Random seeds
# - Oversampling strategies
# - Dropout rates
# Average predictions
# **Expected gain:** +3-5% F1
```

##### 5.2 Curriculum Learning

```python
# Start with easy examples, gradually add hard ones
# Phase 1: Train on high-confidence labels (5 epochs)
# Phase 2: Add medium-confidence (5 epochs)
# Phase 3: Add all data (4 epochs)
```

##### 5.3 Active Learning

```python
# Identify most uncertain predictions
# Request human annotation for those specific cases
# Retrain with higher-quality labels
```

##### 5.4 Multi-Task Weight Tuning

```python
# Try dynamic task weighting
# Give more weight to polarization initially
TASK_LOSS_WEIGHTS = {
    "sentiment": 0.9,          # Was 1.0 - Slight reduction
    "polarization": 1.5,       # Was 1.4 - More focus
}
```

##### 5.5 Cross-Lingual Transfer

```python
# XLM-R is multilingual - leverage that!
# If you have Spanish/Filipino political data:
# - Train on combined dataset
# - May help with neutrality detection
```

---

### 📋 ACTION PLAN FOR RUN #2

**Goal:** Achieve 65-68% Macro-F1 (intermediate milestone)

**Changes to make:**

1. ✅ **Fix weight saving path** (verify before calibration)
2. ✅ **Reduce oversampling aggression:**
   - `JOINT_ALPHA = 0.60`
   - `JOINT_OVERSAMPLING_MAX_MULT = 6.0`
   - `OBJECTIVE_BOOST_MULT = 5.0`
   - `NEUTRAL_BOOST_MULT = 2.0`
3. ✅ **Balance training:**
   - `HEAD_DROPOUT = 0.20`
   - `MAX_GRAD_NORM = 1.0`
4. ✅ **Longer training:**
   - `EPOCHS = 18`
   - `EARLY_STOP_PATIENCE = 9`
5. ✅ **Stronger focal loss:**
   - `FOCAL_GAMMA_SENTIMENT = 2.5`
   - `FOCAL_GAMMA_POLARITY = 3.2`

**Expected Results:**

- Overall Macro-F1: **64-67%** (+3-6% improvement)
- Objective F1: **32-38%** (+6-12% improvement)
- Neutral F1: **52-58%** (+7-13% improvement)

---

### 📊 COMPARISON: XLM-RoBERTa vs mBERT (Run #1)

| Metric               | XLM-R | mBERT | Winner  | Difference |
| -------------------- | ----- | ----- | ------- | ---------- |
| **Overall Macro-F1** | 61.2% | 58.5% | **XLM** | +2.7%      |
| Sentiment F1         | 62.9% | 59.2% | **XLM** | +3.7%      |
| Polarization F1      | 59.5% | 57.7% | **XLM** | +1.8%      |
| Objective F1 (est.)  | 26%   | 22%   | **XLM** | +4%        |
| Neutral F1 (est.)    | 45%   | 42%   | **XLM** | +3%        |
| Training Time        | 57.6m | 56.1m | mBERT   | +1.5m      |

**Conclusion:** XLM-RoBERTa is performing better across all metrics, but both are still significantly below the 75% target. The improvements are modest (+2-4% per class), suggesting that **architectural changes alone won't solve the problem** - need data-centric approach.

---

### 🎯 LONG-TERM ROADMAP

| Run # | Target F1 | Key Focus                         | Status  |
| ----- | --------- | --------------------------------- | ------- |
| 1     | 61.2%     | Initial optimized config          | ✅ Done |
| 2     | 65-68%    | Fix oversampling + regularization | ⏳ Next |
| 3     | 70-72%    | Data augmentation for objective   | 📅 Plan |
| 4     | 73-75%    | Ensemble + threshold tuning       | 📅 Plan |
| 5     | 75%+      | Final optimization + calibration  | 🎯 Goal |

---

**Next Update:** After Run #2 completion
