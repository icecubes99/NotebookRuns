# 🚀 QUICK START: Data Augmentation

**Goal:** Boost XLM-RoBERTa from 68% to 75% macro-F1 in 3-7 days  
**Method:** Augment weak classes (Objective, Neutral)

---

## ⚡ FASTEST PATH (3 DAYS)

### **Step 1: Install Dependencies (5 minutes)**

```bash
pip install googletrans==4.0.0-rc1
pip install sentence-transformers
pip install nlpaug
pip install transformers torch
pip install tqdm pandas numpy
```

### **Step 2: Prepare Your Data (10 minutes)**

Create a CSV with your training data in this format:

```csv
text,sentiment,polarization
"According to the study...",neutral,objective
"I love this policy!",positive,partisan
"This is concerning...",negative,partisan
```

### **Step 3: Run Augmentation (4-6 hours, mostly automated)**

```python
from data_augmentation_toolkit import DataAugmentationPipeline
import pandas as pd

# Load your data
df = pd.read_csv('your_training_data.csv')

# Initialize pipeline (use fast methods only)
pipeline = DataAugmentationPipeline(
    use_backtranslation=True,   # ✅ Enabled (high ROI)
    use_paraphrasing=False,     # ❌ Disabled (slow)
    use_eda=True,               # ✅ Enabled (fast)
    quality_threshold=0.75      # Keep samples 75%+ similar to originals
)

# Augment objective class (highest priority)
df = pipeline.augment_class(
    df=df,
    class_column='polarization',
    class_value='objective',
    text_column='text',
    target_multiplier=5  # 5x more objective samples
)

# Augment neutral class (second priority)
df = pipeline.augment_class(
    df=df,
    class_column='sentiment',
    class_value='neutral',
    text_column='text',
    target_multiplier=3  # 3x more neutral samples
)

# Save augmented dataset
df.to_csv('augmented_training_data.csv', index=False)
```

**⏱️ Expected Runtime:** 4-6 hours (can run overnight)

### **Step 4: Update Training Config (15 minutes)**

In your `XLM_ROBERTA_TRAINING.ipynb`, update:

```python
# === NEW DATA PATH ===
CSV_PATH = '/content/augmented_training_data.csv'

# === REDUCE OVERSAMPLING (no longer needed!) ===
OBJECTIVE_BOOST_MULT = 1.0  # Was 3.5, now have enough data
NEUTRAL_BOOST_MULT = 1.0    # Was 0.3, now have enough data

# === REDUCE CLASS WEIGHTS ===
CLASS_WEIGHT_MULT = {
    "sentiment": {
        "negative": 1.05,
        "neutral":  1.20,    # Reduced from 1.70
        "positive": 1.35
    },
    "polarization": {
        "non_polarized": 1.25,
        "objective":     1.30,  # Reduced from 2.80
        "partisan":      0.90
    }
}

# === ADJUST TRAINING PARAMS ===
EPOCHS = 15              # Reduced from 20 (faster learning with more data)
BATCH_SIZE = 24          # Increased from 16 (have more data)
EARLY_STOP_PATIENCE = 5  # Reduced from 6 (converge faster)
```

### **Step 5: Train Run #12 (1.5 hours)**

Run your training notebook with the augmented data!

**Expected Results:**

- **Objective F1:** 50% → **65-70%** (+15-20%)
- **Neutral F1:** 55% → **68-72%** (+13-17%)
- **Overall Macro-F1:** 68% → **73-76%** (+5-8%)
- **🎯 Target: 75%** ✅ **ACHIEVABLE!**

---

## 📊 WHAT YOU'LL GET

### **Before Augmentation:**

| Class         | Samples | F1  | Issues                    |
| ------------- | ------- | --- | ------------------------- |
| Objective     | 90      | 50% | Too small, ±7-8% variance |
| Neutral       | 401     | 56% | Poor precision (61%)      |
| Non-polarized | 435     | 65% | Below target              |
| Positive      | 208     | 73% | Below target              |

### **After Augmentation:**

| Class         | Samples | Expected F1 | Improvement |
| ------------- | ------- | ----------- | ----------- |
| Objective     | 450+    | 65-70%      | +15-20% 🚀  |
| Neutral       | 1203+   | 68-72%      | +13-17% 🚀  |
| Non-polarized | 435     | 70-73%      | +5-8% ✅    |
| Positive      | 208     | 76-78%      | +3-5% ✅    |

**Overall:** 68% → **73-76%** macro-F1

---

## 🔧 TROUBLESHOOTING

### **Problem: "googletrans" import error**

```bash
# Uninstall old version
pip uninstall googletrans -y

# Install specific working version
pip install googletrans==4.0.0-rc1
```

### **Problem: "sentence-transformers" too slow**

Use a smaller model:

```python
from data_augmentation_toolkit import QualityFilter

# Use faster model
quality_filter = QualityFilter(
    model_name='sentence-transformers/paraphrase-MiniLM-L6-v2'  # Faster!
)
```

### **Problem: "Out of memory" during augmentation**

Process in smaller batches:

```python
# Split your data into chunks
chunk_size = 50  # Process 50 samples at a time
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    augmented_chunk = pipeline.augment_class(chunk, ...)
    # Save each chunk
    augmented_chunk.to_csv(f'augmented_chunk_{i}.csv', index=False)

# Combine all chunks at the end
```

### **Problem: Back-translation is too slow**

**Option 1:** Use only 1-2 languages instead of 4:

```python
from data_augmentation_toolkit import BackTranslationAugmenter

backtrans = BackTranslationAugmenter(
    intermediate_langs=['es', 'fr']  # Just 2 languages
)
```

**Option 2:** Skip back-translation, use only EDA:

```python
pipeline = DataAugmentationPipeline(
    use_backtranslation=False,  # ❌ Disabled
    use_eda=True                # ✅ Only use EDA (fastest)
)
```

---

## 📈 ALTERNATIVE: SUPER FAST (1 DAY)

If you need results **TODAY**, use only EDA:

```python
from data_augmentation_toolkit import EasyDataAugmenter
import pandas as pd

# Load data
df = pd.read_csv('your_training_data.csv')

# Initialize EDA only (very fast!)
eda = EasyDataAugmenter()

# Augment objective class
objective_texts = df[df['polarization'] == 'objective']['text'].tolist()
augmented_objective = eda.augment_batch(objective_texts)

# Create augmented dataframe
aug_df = pd.DataFrame({
    'text': augmented_objective,
    'polarization': 'objective',
    'sentiment': 'neutral',  # Most objectives are neutral
    'is_augmented': True
})

# Combine and save
df['is_augmented'] = False
combined = pd.concat([df, aug_df], ignore_index=True)
combined.to_csv('augmented_fast.csv', index=False)
```

**Expected Results:** 68% → **71-73%** macro-F1 (+3-5%)  
**Time:** 1-2 hours total

---

## 🎯 DECISION MATRIX

| Timeline | Methods          | Expected F1 | Complexity  | Recommended For    |
| -------- | ---------------- | ----------- | ----------- | ------------------ |
| 1 day    | EDA only         | 71-73%      | ⭐ Low      | Need results TODAY |
| 3 days   | Back-Trans + EDA | 73-76%      | ⭐⭐ Med    | **RECOMMENDED** ✅ |
| 7 days   | All + Manual     | 75-78%      | ⭐⭐⭐ High | Maximum quality    |

---

## ✅ CHECKLIST

**Before starting:**

- [ ] Dependencies installed (`pip install ...`)
- [ ] Training data exported as CSV
- [ ] Data has correct columns (`text`, `sentiment`, `polarization`)

**During augmentation:**

- [ ] Augmentation script running (can take 4-6 hours)
- [ ] Monitor console for errors
- [ ] Check intermediate outputs

**After augmentation:**

- [ ] Review `augmented_dataset.csv` (spot check some samples)
- [ ] Update training config (reduce class weights, oversampling)
- [ ] Run training with augmented data
- [ ] Compare Run #12 results to Run #11

**Success criteria:**

- [ ] Objective F1 > 65% ✅
- [ ] Neutral F1 > 68% ✅
- [ ] Overall macro-F1 > 73% ✅

---

## 🚀 START NOW

1. **Right now (5 min):** Install dependencies
2. **Today (4-6 hours):** Run augmentation script (can leave overnight)
3. **Tomorrow (2 hours):** Update config + train Run #12
4. **Result:** 73-76% macro-F1! 🎉

---

## 📞 NEED HELP?

Common questions:

**Q: What if augmented data has wrong labels?**  
A: The quality filter keeps only samples 75%+ similar to originals. Manual validation of 10% sample recommended.

**Q: Will this work for multilingual data?**  
A: Yes! Back-translation and XLM-RoBERTa embeddings work great for multilingual text.

**Q: How much improvement can I expect?**  
A: Realistic expectation: +5-8% macro-F1 (68% → 73-76%)

**Q: What if I don't have time for augmentation?**  
A: Use Option B from Run #12 analysis - tune existing config (warmup ratio, dropout). Expected: +1-2%.

---

## 🎉 READY TO START?

```bash
# Install everything
pip install googletrans==4.0.0-rc1 sentence-transformers nlpaug transformers torch tqdm pandas

# Run the toolkit
python data_augmentation_toolkit.py

# Update your notebook and train!
```

**You got this! 🚀**
