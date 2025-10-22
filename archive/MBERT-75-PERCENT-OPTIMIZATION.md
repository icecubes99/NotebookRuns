# 🎯 mBERT Optimization Strategy: Reaching 75%+ Macro-F1

## 📊 Current vs Target Performance

### **Current Performance (Baseline)**

```
Sentiment Classification:
├─ negative:  73.3% F1 ✅ (Good - 886 samples)
├─ neutral:   49.4% F1 ❌ (CRITICAL - 401 samples)
└─ positive:  62.5% F1 ⚠️  (Needs work - 208 samples)
   → Average: 61.7% F1

Polarization Classification:
├─ non_polarized: 61.6% F1 ⚠️  (435 samples)
├─ objective:     40.4% F1 ❌ (CRITICAL - 90 samples)
└─ partisan:      76.7% F1 ✅ (970 samples)
   → Average: 59.6% F1

OVERALL MACRO-F1: 60.7%
```

### **Target Performance**

```
ALL CLASSES: >75% F1 (STABLE & CONSISTENT)
Overall Macro-F1: 75-80%

Required Improvements:
├─ Sentiment:     +13.3% (61.7% → 75%)
├─ Polarization:  +15.4% (59.6% → 75%)
└─ Overall:       +14.3% (60.7% → 75%)
```

---

## 🔍 ROOT CAUSE ANALYSIS

### **Critical Bottlenecks**

#### **1. Neutral Class (49.4% F1) - BIGGEST PROBLEM**

**Why it fails:**

- Subjective boundaries with negative/positive
- Under-weighted (currently 1.0x multiplier)
- Insufficient training focus
- Middle-sized class (401 samples) gets "squeezed" between larger classes

**Impact:** Loses 10+ percentage points on overall F1

#### **2. Objective Class (40.4% F1) - SECOND BIGGEST**

**Why it fails:**

- Severe class imbalance (only 6% of data)
- Current oversampling (2.5x) insufficient
- Class weight (6.0 capped) not aggressive enough
- Political "objective" is nuanced concept

**Impact:** Loses 8+ percentage points on overall F1

#### **3. Insufficient Model Capacity**

- HEAD_HIDDEN=384 → bottleneck for complex multi-task learning
- Only 2 layers in task heads
- Shared encoder might be under-optimized for diverse tasks

#### **4. Limited Training**

- Only 6 epochs → barely scratching the surface
- Early stopping (patience=3) kicks in too early
- Model hasn't converged fully

#### **5. Suboptimal Class Balancing**

- Current weights: neutral=1.0, objective=1.5 (too conservative)
- Oversampling max=4.0x (should be higher for objective)
- JOINT_ALPHA=0.5 could be more aggressive

---

## 🚀 10-POINT OPTIMIZATION STRATEGY

### **TIER 1: CRITICAL IMPROVEMENTS** (Expected: +8-10% F1)

#### **1. AGGRESSIVE CLASS REWEIGHTING**

```python
# Current (Conservative):
CLASS_WEIGHT_MULT = {
    "sentiment": {"negative": 1.20, "neutral": 1.00, "positive": 1.15},
    "polarization": {"non_polarized": 1.00, "objective": 1.50, "partisan": 1.00}
}

# NEW (Aggressive for 75%+):
CLASS_WEIGHT_MULT = {
    "sentiment": {"negative": 1.10, "neutral": 1.80, "positive": 1.30},  # 🔥 Boost neutral heavily
    "polarization": {"non_polarized": 1.20, "objective": 2.50, "partisan": 0.95}  # 🔥 Massive objective boost
}
MAX_CLASS_WEIGHT = 10.0  # 🔥 Allow higher weights (was 6.0)
```

**Expected Gain:** +4-6% F1

#### **2. EXTREME OVERSAMPLING FOR WEAK CLASSES**

```python
# Current:
JOINT_ALPHA = 0.50
JOINT_OVERSAMPLING_MAX_MULT = 4.0
OBJECTIVE_BOOST_MULT = 2.5

# NEW:
JOINT_ALPHA = 0.70  # 🔥 Much more aggressive joint balancing
JOINT_OVERSAMPLING_MAX_MULT = 8.0  # 🔥 Allow 8x oversampling
OBJECTIVE_BOOST_MULT = 6.0  # 🔥 Triple the objective boost
NEUTRAL_BOOST_MULT = 2.5  # 🔥 NEW: Add neutral-specific boost
USE_STRATIFIED_OVERSAMPLING = True  # 🔥 NEW: Per-class stratification
```

**Expected Gain:** +3-4% F1

#### **3. DOUBLE TRAINING DURATION**

```python
# Current:
EPOCHS = 6
EARLY_STOP_PATIENCE = 3

# NEW:
EPOCHS = 12  # 🔥 Double the epochs
EARLY_STOP_PATIENCE = 6  # 🔥 Much more patience
USE_COSINE_SCHEDULE = True  # 🔥 NEW: Better LR scheduling
LR_MIN = 1e-7  # 🔥 NEW: Don't drop LR to zero
```

**Expected Gain:** +2-3% F1

---

### **TIER 2: ARCHITECTURE ENHANCEMENTS** (Expected: +3-5% F1)

#### **4. LARGER MODEL CAPACITY**

```python
# Current:
HEAD_HIDDEN = 384
HEAD_LAYERS = 2

# NEW:
HEAD_HIDDEN = 768  # 🔥 Double the capacity (match BERT hidden size)
HEAD_LAYERS = 3    # 🔥 Deeper task-specific heads
HEAD_INTERMEDIATE_SIZE = 1024  # 🔥 NEW: Intermediate bottleneck
USE_RESIDUAL_CONNECTIONS = True  # 🔥 NEW: Skip connections in heads
```

**Expected Gain:** +2-3% F1

#### **5. ADVANCED POOLING**

```python
# Current:
REP_POOLING = "last4_mean"

# NEW:
REP_POOLING = "attention_weighted"  # 🔥 Learnable attention pooling
USE_CLS_TOKEN = True  # 🔥 Also use CLS
POOL_CONCAT = True  # 🔥 Concat multiple pooling strategies
```

**Expected Gain:** +1-2% F1

---

### **TIER 3: ADVANCED TECHNIQUES** (Expected: +2-4% F1)

#### **6. MORE AGGRESSIVE FOCAL LOSS**

```python
# Current:
FOCAL_GAMMA_SENTIMENT = 1.0
FOCAL_GAMMA_POLARITY = 1.5

# NEW:
FOCAL_GAMMA_SENTIMENT = 2.0  # 🔥 More focus on hard examples
FOCAL_GAMMA_POLARITY = 2.5   # 🔥 Even more for polarization
USE_ADAPTIVE_GAMMA = True    # 🔥 NEW: Per-class adaptive gamma
```

**Expected Gain:** +1-2% F1

#### **7. ENHANCED REGULARIZATION**

```python
# Current:
RDROP_ALPHA = 0.4
RDROP_WARMUP_EPOCHS = 1

# NEW:
RDROP_ALPHA = 0.6  # 🔥 Stronger consistency loss
RDROP_WARMUP_EPOCHS = 2  # 🔥 Longer warmup
USE_MIXUP = True  # 🔥 NEW: Embedding-level mixup
MIXUP_ALPHA = 0.2  # 🔥 Mild mixup
USE_LABEL_SMOOTHING_ADAPTIVE = True  # 🔥 NEW: Per-class smoothing
```

**Expected Gain:** +1-2% F1

#### **8. MULTI-SAMPLE DROPOUT**

```python
# NEW TECHNIQUE:
USE_MULTI_SAMPLE_DROPOUT = True  # 🔥 Multiple forward passes
NUM_DROPOUT_SAMPLES = 3  # 🔥 3 samples at inference
DROPOUT_INFERENCE_MODE = "vote"  # 🔥 Majority voting
```

**Expected Gain:** +1% F1

---

### **TIER 4: TRAINING OPTIMIZATION** (Expected: +1-2% F1)

#### **9. BETTER LEARNING RATE STRATEGY**

```python
# Current:
LR = 1.5e-5
WARMUP_RATIO = 0.10

# NEW:
LR = 2.5e-5  # 🔥 Higher initial LR (we have more epochs)
WARMUP_RATIO = 0.20  # 🔥 Longer warmup (20% of 12 epochs = 2.4 epochs)
USE_COSINE_ANNEALING = True  # 🔥 Smooth LR decay
LR_SCHEDULER = "cosine_with_restarts"  # 🔥 NEW: Periodic restarts
NUM_CYCLES = 2  # 🔥 2 restart cycles
```

**Expected Gain:** +0.5-1% F1

#### **10. GRADIENT OPTIMIZATION**

```python
# Current:
BATCH_SIZE = 12
GRAD_ACCUM_STEPS = 3  # Effective: 36

# NEW:
BATCH_SIZE = 16  # 🔥 Larger batches
GRAD_ACCUM_STEPS = 3  # Effective: 48 (more stable)
MAX_GRAD_NORM = 0.5  # 🔥 Tighter clipping
USE_GRADIENT_CENTRALIZATION = True  # 🔥 NEW: Better gradient flow
```

**Expected Gain:** +0.5-1% F1

---

## 📊 EXPECTED RESULTS

### **Cumulative Impact**

```
Tier 1 (Critical):     +8-10% F1
Tier 2 (Architecture): +3-5%  F1
Tier 3 (Advanced):     +2-4%  F1
Tier 4 (Optimization): +1-2%  F1
────────────────────────────────
TOTAL POTENTIAL:       +14-21% F1

Current:  60.7% macro-F1
Target:   75.0% macro-F1 (need +14.3%)
Expected: 75-82% macro-F1 ✅ TARGET ACHIEVABLE!
```

### **Per-Class Projections**

```
Sentiment:
├─ negative:  73% → 78-82% (+5-9%)
├─ neutral:   49% → 70-76% (+21-27%) ← Biggest gain
└─ positive:  63% → 75-80% (+12-17%)
   → Average: 62% → 74-79% ✅

Polarization:
├─ non_polarized: 62% → 73-78% (+11-16%)
├─ objective:     40% → 68-75% (+28-35%) ← Biggest gain
└─ partisan:      77% → 80-85% (+3-8%)
   → Average: 60% → 74-79% ✅
```

---

## ⏱️ TRAINING TIME IMPACT

```
Current: ~36-40 minutes
New:     ~55-70 minutes

Breakdown:
├─ More epochs (6→12):        +30 min
├─ Larger model (384→768):    +8 min
├─ Larger batches (12→16):    -5 min
├─ More samples (oversampling): +7 min
└─ Advanced techniques:       +5 min
────────────────────────────────────
TOTAL: ~55-70 minutes

Worth it? YES - for +14-21% F1 improvement!
```

---

## 🎯 IMPLEMENTATION PRIORITY

### **Phase 1: Quick Wins** (15 min to implement)

1. ✅ Increase class weights (neutral=1.8, objective=2.5)
2. ✅ Boost oversampling (6x objective, 2.5x neutral)
3. ✅ Double epochs (6→12)
4. ✅ Increase patience (3→6)

**Expected: 60.7% → 68-70% F1** (+7-9% gain)

### **Phase 2: Architecture** (20 min to implement)

5. ✅ Larger heads (384→768)
6. ✅ Deeper heads (2→3 layers)
7. ✅ Better pooling (attention-weighted)

**Expected: 68-70% → 72-74% F1** (+4-4% gain)

### **Phase 3: Advanced** (25 min to implement)

8. ✅ Stronger focal loss (gamma 2.0/2.5)
9. ✅ Mixup + enhanced R-Drop
10. ✅ Cosine LR schedule

**Expected: 72-74% → 75-78% F1** (+3-4% gain)

---

## ✅ SUCCESS METRICS

### **Minimum Acceptable Performance (75% threshold)**

```
✅ Sentiment F1:     ≥ 75%
✅ Polarization F1:  ≥ 75%
✅ Overall Macro-F1: ≥ 75%
✅ Weakest Class F1: ≥ 68% (neutral or objective)
✅ Training Stability: No NaN losses, consistent improvement
```

### **Stretch Goal (80%+ performance)**

```
🎯 Sentiment F1:     ≥ 78%
🎯 Polarization F1:  ≥ 77%
🎯 Overall Macro-F1: ≥ 77.5%
🎯 All Classes F1:   ≥ 70%
```

---

## 🚨 RISKS & MITIGATION

### **Risk 1: Overfitting**

**Mitigation:**

- Strong regularization (weight decay 0.03)
- R-Drop + Mixup
- Label smoothing
- Validation-based early stopping

### **Risk 2: Training Instability**

**Mitigation:**

- Gradient clipping (0.5)
- Longer warmup (20%)
- Capped class weights (10.0 max)
- Monitoring loss curves

### **Risk 3: Class Imbalance Worse**

**Mitigation:**

- Extreme oversampling (8x max)
- Per-class focal loss
- Adaptive class weights
- Validation on balanced metrics

---

## 📝 NEXT STEPS

1. **Implement Phase 1** (Quick Wins) - Test immediately
2. **Monitor Results** - Check if we hit 68-70% F1
3. **Add Phase 2** (Architecture) - Push to 72-74% F1
4. **Fine-tune Phase 3** (Advanced) - Reach 75-78% F1
5. **Iterate** - If needed, add ensemble/pseudo-labeling

**LET'S PUSH mBERT TO 75%+!** 🚀
