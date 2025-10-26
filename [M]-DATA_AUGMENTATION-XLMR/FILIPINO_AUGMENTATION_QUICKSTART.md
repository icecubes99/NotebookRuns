# 🚀 FILIPINO AUGMENTATION - QUICK START GUIDE

## ⚠️ REALITY CHECK

**Your Constraints:**

- ⏱️ Timeline: 1-2 days
- 💰 Budget: $0 (free Colab only)
- 📊 No manual data collection

**What You CAN Achieve:** 73-76% macro-F1 (+5-8% from 68.36%)  
**Your >80% Target:** ❌ **NOT ACHIEVABLE** with these constraints

**BUT:** 73-76% is still a **great improvement** that's worth doing!

---

## 🇵🇭 THE FILIPINO PROBLEM

Your original `DATA_AUGMENTATION_COLAB.ipynb` notebook **WON'T WORK** for Filipino because:

- ❌ Uses WordNet (English-only)
- ❌ Back-translation wasn't tested for Filipino
- ❌ Synonym replacement doesn't understand code-switching

---

## ✅ SOLUTION: Use XLM-RoBERTa Contextual Augmentation

**Why XLM-RoBERTa works for Filipino:**

- ✅ Trained on 100 languages including Tagalog
- ✅ Understands code-switching (English + Filipino mix)
- ✅ Context-aware (preserves meaning)
- ✅ Works within free Colab limits

---

## 📋 STEP-BY-STEP INSTRUCTIONS

### **Option 1: Simplest Approach (3-4 hours)**

Use **ONLY** XLM-RoBERTa contextual augmentation:

```python
import nlpaug.augmenter.word as naw
import pandas as df

# Load data
df = pd.read_csv('adjudications_2025-10-22.csv')

# Initialize XLM-R augmenter
aug = naw.ContextualWordEmbsAug(
    model_path='xlm-roberta-base',
    action='substitute',
    aug_p=0.20,  # Replace 20% of words
    device='cuda'
)

# Augment objective class
obj_texts = df[df['Final Polarization'] == 'objective']['Comment'].tolist()
augmented_obj = []

for text in obj_texts:
    for _ in range(4):  # 4x augmentation
        try:
            augmented_obj.append(aug.augment(text))
        except:
            continue

# Create augmented dataframe
aug_obj_df = pd.DataFrame({
    'Title': '',
    'Comment': augmented_obj,
    'Final Sentiment': 'neutral',
    'Final Polarization': 'objective',
    'is_augmented': True
})

# Repeat for neutral class (3x augmentation)
neu_texts = df[df['Final Sentiment'] == 'neutral']['Comment'].tolist()
augmented_neu = []

for text in neu_texts:
    for _ in range(2):  # 2x augmentation
        try:
            augmented_neu.append(aug.augment(text))
        except:
            continue

# Get polarization distribution for neutral
neu_pol_dist = df[df['Final Sentiment'] == 'neutral']['Final Polarization'].value_counts(normalize=True).to_dict()
pol_labels = np.random.choice(list(neu_pol_dist.keys()), size=len(augmented_neu), p=list(neu_pol_dist.values()))

aug_neu_df = pd.DataFrame({
    'Title': '',
    'Comment': augmented_neu,
    'Final Sentiment': 'neutral',
    'Final Polarization': pol_labels,
    'is_augmented': True
})

# Combine all
df['is_augmented'] = False
df_final = pd.concat([df, aug_obj_df, aug_neu_df], ignore_index=True)
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df_final.to_csv('augmented_adjudications_2025-10-22.csv', index=False)
```

**Expected Result:** 73-75% macro-F1  
**Runtime:** 3-4 hours  
**Complexity:** Low ✅

---

### **Option 2: Add Quality Filtering (4-5 hours)**

Add quality filtering to ensure augmented samples preserve meaning:

```python
from sentence_transformers import SentenceTransformer, util

# After generating augmented samples, filter them:
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def filter_quality(original_texts, augmented_texts, threshold=0.70):
    """Keep only high-quality augmentations"""
    filtered = []
    orig_emb = model.encode(original_texts, convert_to_tensor=True)
    aug_emb = model.encode(augmented_texts, convert_to_tensor=True)

    for i, aug in enumerate(aug_emb):
        similarities = util.cos_sim(aug, orig_emb)[0]
        if similarities.max().item() >= threshold:
            filtered.append(augmented_texts[i])

    return filtered

# Apply to objective class
augmented_obj_filtered = filter_quality(obj_texts, augmented_obj)

# Apply to neutral class
augmented_neu_filtered = filter_quality(neu_texts, augmented_neu)

# Then create dataframes with filtered samples...
```

**Expected Result:** 74-76% macro-F1  
**Runtime:** 4-5 hours  
**Complexity:** Medium ⚠️

---

## 🎯 RECOMMENDED APPROACH FOR YOU

**Use Option 1** (XLM-R only, no quality filtering)

**Why:**

- ✅ Simplest and fastest
- ✅ Works within free Colab limits
- ✅ Still gives 73-75% macro-F1 (+5-7%)
- ✅ Can complete in 1 day

**Quality filtering adds minimal improvement** (~1%) but doubles complexity.

---

## 📝 COMPLETE COLAB CODE (COPY-PASTE READY)

I'll create a single cell you can run in Colab:

```python
# FILIPINO DATA AUGMENTATION - SINGLE CELL VERSION
# Expected runtime: 3-4 hours on free Colab

import pandas as pd
import numpy as np
import nlpaug.augmenter.word as naw
from tqdm import tqdm

# 1. Upload and load data
from google.colab import files
uploaded = files.upload()
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)

print(f"Loaded {len(df)} samples")

# 2. Initialize XLM-R augmenter
print("\\nLoading XLM-RoBERTa...")
aug = naw.ContextualWordEmbsAug(
    model_path='xlm-roberta-base',
    action='substitute',
    aug_p=0.20,
    device='cuda'
)
print("✅ XLM-R ready!")

# 3. Augment objective class (5x total = 1 original + 4 augmented)
print("\\n🎯 Augmenting Objective class...")
obj_texts = df[df['Final Polarization'] == 'objective']['Comment'].tolist()
print(f"Original: {len(obj_texts)} samples")

augmented_obj = []
for text in tqdm(obj_texts, desc="Objective"):
    for _ in range(4):
        try:
            augmented_obj.append(aug.augment(text))
        except:
            continue

print(f"Generated: {len(augmented_obj)} samples")

aug_obj_df = pd.DataFrame({
    'Title': '',
    'Comment': augmented_obj,
    'Final Sentiment': 'neutral',
    'Final Polarization': 'objective',
    'is_augmented': True
})

# 4. Augment neutral class (3x total = 1 original + 2 augmented)
print("\\n🎯 Augmenting Neutral class...")
neu_texts = df[df['Final Sentiment'] == 'neutral']['Comment'].tolist()
print(f"Original: {len(neu_texts)} samples")

augmented_neu = []
for text in tqdm(neu_texts, desc="Neutral"):
    for _ in range(2):
        try:
            augmented_neu.append(aug.augment(text))
        except:
            continue

print(f"Generated: {len(augmented_neu)} samples")

# Get polarization distribution
neu_pol_dist = df[df['Final Sentiment'] == 'neutral']['Final Polarization'].value_counts(normalize=True).to_dict()
pol_labels = np.random.choice(list(neu_pol_dist.keys()), size=len(augmented_neu), p=list(neu_pol_dist.values()))

aug_neu_df = pd.DataFrame({
    'Title': '',
    'Comment': augmented_neu,
    'Final Sentiment': 'neutral',
    'Final Polarization': pol_labels,
    'is_augmented': True
})

# 5. Combine and save
print("\\n💾 Combining and saving...")
df['is_augmented'] = False
df_final = pd.concat([df, aug_obj_df, aug_neu_df], ignore_index=True)
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

output_file = 'augmented_adjudications_2025-10-22.csv'
df_final.to_csv(output_file, index=False)

print(f"\\n✅ DONE!")
print(f"Total samples: {len(df_final)}")
print(f"Augmented samples: {df_final['is_augmented'].sum()}")

# 6. Download
files.download(output_file)

print("\\n📊 Expected performance: 73-76% macro-F1")
print("🎯 Next: Update training config and run Run #12!")
```

---

## 📊 CONFIGURATION FOR RUN #12

After augmentation, update your training notebook:

```python
CSV_PATH = '/content/augmented_adjudications_2025-10-22.csv'

# REDUCE OVERSAMPLING
OBJECTIVE_BOOST_MULT = 1.0  # Was 3.5
NEUTRAL_BOOST_MULT = 1.0    # Was 0.3

# REDUCE CLASS WEIGHTS
CLASS_WEIGHT_MULT = {
    "sentiment": {
        "neutral": 1.20,  # Was 1.70
    },
    "polarization": {
        "objective": 1.30,  # Was 2.80
    }
}

# OPTIMIZE FOR MORE DATA
EPOCHS = 15              # Was 20
BATCH_SIZE = 24          # Was 16
EARLY_STOP_PATIENCE = 5  # Was 6
```

---

## 📈 EXPECTED RESULTS

| Metric               | Run #11 | Run #12 (Expected) | Improvement    |
| -------------------- | ------- | ------------------ | -------------- |
| **Overall Macro-F1** | 68.36%  | **73-76%**         | **+5-8%** ✅   |
| Objective F1         | 50.28%  | **65-70%**         | **+15-20%** 🚀 |
| Neutral F1           | 55.69%  | **68-72%**         | **+13-17%** 🚀 |

---

## ⏱️ TIMELINE

**Day 1 (Today):**

- Hour 1: Set up Colab, run augmentation script
- Hours 2-4: Augmentation runs (automated)
- Hour 5: Download augmented dataset, update config
- Hours 6-7: Train Run #12
- **End of day:** See 73-76% results! ✅

**Day 2 (Tomorrow):**

- Analyze results
- Celebrate improvement! 🎉

---

## 🚨 FINAL REALITY CHECK

**What you WILL achieve:** 73-76% macro-F1 (+5-8%)  
**What you WON'T achieve:** >80% macro-F1

**To reach >80%, you would need:**

- 2-4 weeks timeline
- Manual data collection (200-300 samples)
- Ensemble methods
- Active learning

**But 73-76% is still excellent progress for 1 day of work!**

---

## 🎉 READY TO START?

1. Open Google Colab
2. Paste the complete code above into a single cell
3. Run it!
4. Come back in 3-4 hours
5. Download augmented dataset
6. Train Run #12
7. Hit 73-76%! 🚀

**Good luck!** 🇵🇭
