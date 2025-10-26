# 📊 DATA AUGMENTATION STRATEGY FOR XLM-RoBERTa

## Fast Path to 75% Macro-F1

**Current Status:** 68-69% baseline (optimization ceiling reached)  
**Target:** 75% macro-F1  
**Gap:** ~6-7%  
**Strategy:** Data augmentation for weak classes

---

## 🎯 PRIORITY TARGETS

### **Critical Classes (Need Data Urgently)**

| Class         | Current F1 | Target F1 | Gap    | Samples | Priority  | Impact Potential |
| ------------- | ---------- | --------- | ------ | ------- | --------- | ---------------- |
| **Objective** | 50.28%     | 75%       | -24.7% | 90      | 🔴 **P0** | **+10-15% F1**   |
| **Neutral**   | 55.69%     | 75%       | -19.3% | 401     | 🟠 **P1** | **+5-8% F1**     |
| Non-polarized | 64.85%     | 75%       | -10.2% | 435     | 🟡 **P2** | **+3-5% F1**     |
| Positive      | 72.77%     | 75%       | -2.2%  | 208     | 🟢 **P3** | **+2-3% F1**     |

**Key Insight:**

- **Objective class (90 samples)** is the bottleneck → ±7-8% F1 variance
- **Neutral class (401 samples)** has poor precision (61.89%) → needs clearer examples
- If we can boost these 2 classes, we can reach 72-74% macro-F1

---

## 🚀 FAST IMPLEMENTATION PLAN (7 DAYS)

### **Phase 1: Data Collection & Analysis (Days 1-2)**

#### **Step 1.1: Analyze Existing Data**

```python
# Identify weak samples (misclassified examples)
# These tell us what patterns the model struggles with
```

**Actions:**

1. Export all test set misclassifications for Objective and Neutral classes
2. Analyze confusion patterns:
   - What Objective samples are classified as Non-polarized?
   - What Neutral samples are classified as Negative/Positive?
3. Identify common features in misclassified samples

**Time:** 2-4 hours  
**Outcome:** List of confusing patterns to target

---

#### **Step 1.2: Source Augmentation Targets**

**For Objective Class (90 → 300+ samples):**

**Priority Sources:**

1. ✅ **News Articles (Reporting style)**

   - Reuters, AP, BBC (multilingual)
   - Look for: "reported," "according to," "officials said"
   - Target: 100 new samples

2. ✅ **Wikipedia Introductions**

   - Factual, encyclopedic tone
   - Neutral language, citation-heavy
   - Target: 50 new samples

3. ✅ **Academic Abstracts**

   - Research papers, studies
   - Objective reporting of findings
   - Target: 50 new samples

4. ✅ **Government/Official Statements**
   - Census data, official reports
   - Policy descriptions (non-partisan)
   - Target: 50 new samples

**Expected Objective Class After Augmentation:** 90 → 290 samples (+222%)

---

**For Neutral Class (401 → 700+ samples):**

**Priority Sources:**

1. ✅ **Questions & Queries**

   - "How do I...?", "What is...?", "Where can...?"
   - No sentiment, just information-seeking
   - Target: 150 new samples

2. ✅ **Factual Statements**

   - Weather reports, schedules, facts
   - "The meeting is at 3pm," "It will rain tomorrow"
   - Target: 100 new samples

3. ✅ **Product Descriptions (Neutral)**
   - Technical specifications
   - Feature lists without opinion
   - Target: 50 new samples

**Expected Neutral Class After Augmentation:** 401 → 701 samples (+75%)

---

### **Phase 2: Manual Data Collection (Days 3-4)**

#### **Step 2.1: Create Collection Guidelines**

**Objective Polarization - Collection Criteria:**

```
✅ INCLUDE:
- Factual reporting with no opinion
- Multiple perspectives presented equally
- Citations, data, statistics
- "X said...", "According to...", "Reports show..."
- Academic/professional tone

❌ EXCLUDE:
- Any evaluative language ("good", "bad", "should")
- Emotional words ("outrage", "celebrate", "disaster")
- One-sided arguments
- Persuasive language
```

**Neutral Sentiment - Collection Criteria:**

```
✅ INCLUDE:
- Questions without emotional charge
- Factual statements
- Information requests
- Procedural descriptions
- Technical language

❌ EXCLUDE:
- Complaints or praise
- Emotive language
- Evaluative words
- Frustration or satisfaction indicators
```

---

#### **Step 2.2: Rapid Collection Methods**

**Method 1: Web Scraping (Fastest) - 4 hours**

```python
# Sources:
# - Wikipedia API (multilingual, objective)
# - News APIs (Reuters, AP)
# - Academic databases (PubMed abstracts)
# - Government data portals
```

**Expected Yield:** 200-300 samples

**Method 2: Synthetic Generation (GPT-4) - 2 hours**

```python
# Prompt engineering for:
# - Objective news summaries
# - Neutral information requests
# - Factual descriptions
```

**Expected Yield:** 100-200 samples  
**Note:** Requires manual validation

**Method 3: Crowdsourcing (Parallel) - 8 hours**

```python
# Platform: Amazon MTurk, Figure Eight
# Task: Label existing unlabeled text
# Focus: Political news, social media
```

**Expected Yield:** 300-500 samples  
**Note:** Requires quality control

---

### **Phase 3: Data Augmentation Techniques (Days 3-5)**

#### **Technique 1: Back-Translation (Best for Multilingual)**

**How it works:**

1. Translate text to intermediate language (e.g., English → Spanish)
2. Translate back to original language
3. Creates paraphrased version with same meaning

**Implementation:**

```python
from googletrans import Translator

def back_translate(text, src_lang='en', intermediate_lang='es'):
    translator = Translator()
    # Translate to intermediate language
    intermediate = translator.translate(text, src=src_lang, dest=intermediate_lang).text
    # Translate back
    back = translator.translate(intermediate, src=intermediate_lang, dest=src_lang).text
    return back

# For each Objective/Neutral sample:
# 1. Back-translate through Spanish
# 2. Back-translate through French
# 3. Back-translate through German
# Result: 3x data multiplication
```

**Expected Objective Class:** 290 → 870 samples (+3x)  
**Expected Neutral Class:** 701 → 2,103 samples (+3x)

**Time:** 4-6 hours (automated)  
**Quality:** High (preserves meaning, adds lexical diversity)

---

#### **Technique 2: Synonym Replacement**

**How it works:**
Replace non-critical words with synonyms while preserving sentiment/polarization

**Implementation:**

```python
import nlpaug.augmenter.word as naw

# Use contextual word embeddings (BERT-based)
aug = naw.ContextualWordEmbsAug(
    model_path='xlm-roberta-base',
    action='substitute',
    aug_p=0.15  # Replace 15% of words
)

# For each sample:
augmented_text = aug.augment(original_text)
```

**Expected Multiplication:** 2x (1 original + 1 augmented)

**Time:** 2-3 hours  
**Quality:** Medium-High (preserves meaning)

---

#### **Technique 3: Paraphrasing (GPT-4/T5)**

**How it works:**
Use large language models to generate paraphrases that maintain label

**Implementation:**

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

def paraphrase(text):
    input_text = f"paraphrase: {text}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512)
    outputs = model.generate(inputs, max_length=512, num_return_sequences=3)
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Generate 3 paraphrases per sample
```

**Expected Multiplication:** 3-4x

**Time:** 6-8 hours  
**Quality:** High (human-like paraphrases)

---

#### **Technique 4: EDA (Easy Data Augmentation)**

**4 Simple Operations:**

1. **Synonym replacement** (replace n words with synonyms)
2. **Random insertion** (insert random synonym)
3. **Random swap** (swap two words)
4. **Random deletion** (randomly delete words)

**Implementation:**

```python
import nlpaug.augmenter.word as naw

# Random insertion
aug_insert = naw.SynonymAug(aug_src='wordnet')
# Random swap
aug_swap = naw.RandomWordAug(action='swap')
# Random delete
aug_delete = naw.RandomWordAug(action='delete')

# Apply each technique once per sample
```

**Expected Multiplication:** 3-4x

**Time:** 2-3 hours  
**Quality:** Medium (preserves general meaning)

---

### **Phase 4: Quality Control & Validation (Days 6-7)**

#### **Step 4.1: Automated Quality Checks**

**Check 1: Semantic Similarity (Ensure augmentation preserves meaning)**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

def check_similarity(original, augmented, threshold=0.75):
    emb1 = model.encode([original])
    emb2 = model.encode([augmented])
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return similarity >= threshold  # Keep if similar enough

# Filter: Keep only augmented samples with similarity >= 0.75
```

**Check 2: Label Preservation (Ensure augmentation doesn't flip label)**

```python
# Use trained model to predict on augmented samples
# If prediction confidence < 0.6, flag for manual review
```

**Check 3: Duplicate Detection**

```python
# Use MinHash or SimHash to detect near-duplicates
# Remove if similarity > 0.95 with any existing sample
```

---

#### **Step 4.2: Manual Validation (Critical)**

**Validation Protocol:**

1. Sample 10% of augmented data randomly
2. Manually verify labels are correct
3. If accuracy < 90%, adjust augmentation parameters
4. Iterate until quality threshold met

**Time:** 4-6 hours  
**Critical for:** Avoiding label noise that hurts performance

---

### **Phase 5: Integration & Retraining (Day 7)**

#### **Step 5.1: Update Dataset**

**New Dataset Composition:**
| Class | Original | Collected | Augmented | Total | Increase |
| ------------- | -------- | --------- | --------- | ------ | --------- |
| Objective | 90 | 200 | 870 | 1,160 | **+1189%** |
| Neutral | 401 | 300 | 2,103 | 2,804 | **+599%** |
| Negative | 886 | - | 886 | 1,772 | +100% |
| Positive | 208 | 100 | 312 | 620 | +198% |
| Non-polarized | 435 | 100 | 653 | 1,188 | +173% |
| Partisan | 970 | - | 970 | 1,940 | +100% |

**Key Changes:**

- Objective class: 90 → 1,160 (±7-8% variance → ±1-2% variance expected!)
- Neutral class: 401 → 2,804 (precision should improve significantly)

---

#### **Step 5.2: Adjust Training Configuration for Larger Dataset**

**Update Oversampling (No Longer Needed for Objective/Neutral):**

```python
# OLD:
OBJECTIVE_BOOST_MULT = 3.5  # Was needed for 90 samples
NEUTRAL_BOOST_MULT = 0.3    # Was needed to prevent overfitting

# NEW:
OBJECTIVE_BOOST_MULT = 1.0  # Natural distribution is now good!
NEUTRAL_BOOST_MULT = 1.0    # No longer need to undersample!
```

**Update Class Weights (Less Aggressive):**

```python
# OLD:
CLASS_WEIGHT_MULT = {
    "sentiment": {
        "neutral": 1.70,  # Was needed for 401 samples
    },
    "polarization": {
        "objective": 2.80,  # Was needed for 90 samples
    }
}

# NEW:
CLASS_WEIGHT_MULT = {
    "sentiment": {
        "neutral": 1.20,  # Reduce (now have 2,804 samples)
    },
    "polarization": {
        "objective": 1.30,  # Reduce (now have 1,160 samples)
    }
}
```

**Update Training Parameters:**

```python
# With more data, we can:
EPOCHS = 15  # Reduce from 20 (more data = faster learning)
BATCH_SIZE = 24  # Increase from 16 (have more samples)
EARLY_STOP_PATIENCE = 5  # Reduce from 6 (converge faster)
```

---

#### **Step 5.3: Expected Performance After Augmentation**

**Predicted Results (Run #12 with Augmented Data):**

| Class         | Current F1 | Expected F1 | Improvement | Rationale                         |
| ------------- | ---------- | ----------- | ----------- | --------------------------------- |
| **Objective** | 50.28%     | **65-70%**  | **+15-20%** | 1,160 samples → stable learning   |
| **Neutral**   | 55.69%     | **68-72%**  | **+12-16%** | 2,804 samples → better patterns   |
| Positive      | 72.77%     | **76-78%**  | **+3-5%**   | 620 samples → reach target        |
| Non-polarized | 64.85%     | **70-73%**  | **+5-8%**   | 1,188 samples → better separation |
| Negative      | 83.05%     | **83-85%**  | **+0-2%**   | Already strong, maintain          |
| Partisan      | 83.54%     | **84-86%**  | **+0-2%**   | Already strong, maintain          |

**Overall Macro-F1:**

- Current: 68.36%
- Expected: **73-76%** (+5-8%)
- **Target: 75%** ✅ ACHIEVABLE!

---

## 📋 COMPLETE 7-DAY TIMELINE

| Day | Task                                 | Time    | Outcome                       |
| --- | ------------------------------------ | ------- | ----------------------------- |
| 1   | Analyze misclassifications           | 4 hours | Understand failure patterns   |
| 1-2 | Define collection guidelines         | 2 hours | Clear criteria for new data   |
| 2-3 | Collect new samples (web scraping)   | 8 hours | +200-300 objective/neutral    |
| 3-4 | Apply back-translation (3 languages) | 6 hours | 3x multiplication             |
| 4-5 | Apply paraphrasing (T5/GPT-4)        | 8 hours | 3-4x multiplication           |
| 5-6 | Quality control (automated + manual) | 6 hours | Remove bad augmentations      |
| 6   | Integrate new data + update config   | 4 hours | New training dataset ready    |
| 7   | Retrain model (Run #12)              | 1.5 hrs | Evaluate performance          |
|     | **TOTAL**                            | ~40 hrs | **Expected: 73-76% macro-F1** |

---

## 🛠️ IMPLEMENTATION TOOLKIT

### **Required Libraries**

```bash
pip install nlpaug googletrans==4.0.0-rc1 sentence-transformers transformers
pip install openai  # For GPT-4 paraphrasing (optional)
pip install beautifulsoup4 requests  # For web scraping
```

### **Code Templates**

#### **Template 1: Back-Translation Pipeline**

```python
from googletrans import Translator
import time

def augment_via_backtranslation(texts, intermediate_langs=['es', 'fr', 'de']):
    translator = Translator()
    augmented = []

    for text in texts:
        for lang in intermediate_langs:
            try:
                # Translate to intermediate
                intermediate = translator.translate(text, dest=lang).text
                time.sleep(0.5)  # Rate limiting

                # Translate back
                back = translator.translate(intermediate, dest='en').text
                time.sleep(0.5)

                augmented.append(back)
            except Exception as e:
                print(f"Error: {e}")
                continue

    return augmented

# Usage:
objective_samples = df[df['polarization'] == 'objective']['text'].tolist()
augmented_objective = augment_via_backtranslation(objective_samples)
```

#### **Template 2: Quality Filter**

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

def filter_quality(original_texts, augmented_texts, threshold=0.75):
    """Keep only high-quality augmentations"""
    filtered = []

    for orig, aug in zip(original_texts, augmented_texts):
        # Check semantic similarity
        emb_orig = model.encode(orig, convert_to_tensor=True)
        emb_aug = model.encode(aug, convert_to_tensor=True)
        similarity = util.cos_sim(emb_orig, emb_aug).item()

        if similarity >= threshold:
            filtered.append({
                'original': orig,
                'augmented': aug,
                'similarity': similarity
            })

    return filtered
```

#### **Template 3: Dataset Integration**

```python
import pandas as pd

def integrate_augmented_data(original_df, augmented_samples, class_col, class_value):
    """Add augmented samples to dataset"""
    # Create augmented dataframe
    aug_df = pd.DataFrame({
        'text': augmented_samples,
        class_col: class_value,
        'is_augmented': True  # Track which samples are synthetic
    })

    # Combine with original
    combined = pd.concat([original_df, aug_df], ignore_index=True)

    # Shuffle
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    return combined

# Usage:
df = integrate_augmented_data(
    original_df=train_df,
    augmented_samples=augmented_objective,
    class_col='polarization',
    class_value='objective'
)
```

---

## 🎯 ALTERNATIVE: FAST TRACK (3 DAYS)

If 7 days is too long, here's a compressed plan:

### **Day 1: Synthetic Generation Only**

- Use GPT-4 to generate 200 objective + 300 neutral samples
- Manual validation (4 hours)
- **Expected improvement:** +3-5% macro-F1

### **Day 2: Back-Translation**

- Apply back-translation to all weak class samples
- Automated quality filtering
- **Expected improvement:** +2-4% macro-F1

### **Day 3: Retrain**

- Update configuration
- Train Run #12
- **Total Expected:** 71-73% macro-F1

**Total Time:** 24 hours of active work  
**Result:** 71-73% macro-F1 (close to target!)

---

## 📊 EXPECTED ROI

| Augmentation Method | Time Cost | Sample Increase | Expected F1 Gain | ROI (Gain/Hour)   |
| ------------------- | --------- | --------------- | ---------------- | ----------------- |
| Manual Collection   | 8 hours   | +200-300        | +3-5%            | 0.4-0.6%/hour     |
| Back-Translation    | 6 hours   | 3x data         | +3-5%            | 0.5-0.8%/hour     |
| GPT-4 Paraphrasing  | 8 hours   | 3-4x data       | +4-6%            | 0.5-0.75%/hour    |
| EDA (Simple)        | 3 hours   | 3-4x data       | +2-3%            | 0.7-1.0%/hour     |
| **COMBINED (All)**  | 40 hours  | 10-15x data     | **+8-12%**       | **0.2-0.3%/hour** |

**Recommendation:** Start with **Back-Translation + GPT-4** (fastest + highest quality)

---

## 🚨 RISKS & MITIGATION

### **Risk 1: Label Noise**

**Problem:** Augmented samples may have incorrect labels  
**Mitigation:**

- Always validate 10% manually
- Use confidence thresholding (keep only high-confidence augmentations)
- Track augmented vs. original sample performance separately

### **Risk 2: Overfitting to Augmentation Artifacts**

**Problem:** Model learns augmentation patterns instead of real patterns  
**Mitigation:**

- Use diverse augmentation techniques (not just one)
- Mix augmented data with original (don't oversample augmented)
- Validate on purely original test set

### **Risk 3: Time Investment**

**Problem:** 40 hours is a lot of work  
**Mitigation:**

- Start with fastest methods (back-translation, GPT-4)
- Parallelize (run augmentation scripts overnight)
- Focus on objective class first (highest impact)

---

## 🎉 NEXT STEPS

**Immediate Actions (Today):**

1. ✅ Install required libraries (`pip install nlpaug googletrans sentence-transformers`)
2. ✅ Extract objective and neutral samples from current dataset
3. ✅ Start back-translation pipeline (can run overnight)

**Tomorrow:** 4. ✅ Review augmented samples 5. ✅ Apply quality filtering 6. ✅ Integrate into dataset

**Day 3:** 7. ✅ Update training configuration 8. ✅ Run training (Run #12) 9. ✅ Evaluate results

**Expected Outcome:** 73-76% macro-F1 within 3-7 days!

---

## 📞 READY TO START?

**Quick Decision Matrix:**

| Timeline | Method                           | Expected F1 | Complexity |
| -------- | -------------------------------- | ----------- | ---------- |
| 3 days   | Back-Translation + GPT-4         | 71-73%      | Medium     |
| 7 days   | Full Augmentation Suite          | 73-76%      | High       |
| 14 days  | + Manual Collection + Validation | 75-78%      | Very High  |

**My Recommendation for "Fast":** **3-day plan with Back-Translation + GPT-4**

Want me to create the implementation scripts now? 🚀
