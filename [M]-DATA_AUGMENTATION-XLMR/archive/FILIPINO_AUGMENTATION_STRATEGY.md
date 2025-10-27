# 🇵🇭 Filipino/Taglish Data Augmentation Strategy

## Realistic Path to >80% Macro-F1

**Current Status:** 68.36% baseline (Run #11)  
**Target:** >80% macro-F1  
**Gap:** +11.64% (VERY AMBITIOUS)  
**Reality Check:** Data augmentation alone typically gives +5-8%

---

## 🎯 MULTI-PRONGED APPROACH FOR >80%

To reach >80%, you need **ALL** of these strategies combined:

| Strategy                                  | Expected Gain | Cumulative F1 | Priority |
| ----------------------------------------- | ------------- | ------------- | -------- |
| **Phase 1:** Filipino-aware augmentation  | +5-7%         | 73-75%        | 🔴 P0    |
| **Phase 2:** Manual data collection       | +3-5%         | 76-80%        | 🟠 P1    |
| **Phase 3:** Ensemble models              | +2-4%         | 78-84%        | 🟡 P2    |
| **Phase 4:** Active learning & refinement | +1-3%         | 79-87%        | 🟢 P3    |

**Key Insight:** You need 3-4 phases to reliably hit >80%

---

## 📋 PHASE 1: FILIPINO-AWARE DATA AUGMENTATION

### **1A: Contextual Word Substitution (Best for Filipino)**

Uses XLM-RoBERTa embeddings to find similar words **in context** (language-agnostic!)

```python
import nlpaug.augmenter.word as naw

# Use XLM-RoBERTa for Filipino-aware augmentation
aug = naw.ContextualWordEmbsAug(
    model_path='xlm-roberta-base',  # Already multilingual!
    action='substitute',
    aug_p=0.15,  # Replace 15% of words
    device='cuda'
)

# Example:
text = "Ang gobyerno ay dapat magbigay ng tulong sa mga mahihirap"
augmented = aug.augment(text)
# XLM-R understands Filipino context and substitutes appropriately
```

**Why this works:**

- XLM-RoBERTa was trained on 100 languages including Tagalog
- Understands code-switching (English + Filipino mixed)
- Context-aware substitutions preserve meaning

**Expected gain:** +3-5% macro-F1

---

### **1B: Back-Translation Through Tagalog (Test First!)**

Google Translate's Filipino support varies, so we need to **test quality first**:

```python
from googletrans import Translator

translator = Translator()

# Test sample
text = "Ang gobyerno ay dapat magbigay ng tulong sa mga mahihirap"

# Translate to English
english = translator.translate(text, src='tl', dest='en').text
print(f"English: {english}")

# Translate back to Tagalog
back = translator.translate(english, src='en', dest='tl').text
print(f"Back: {back}")

# MANUAL CHECK: Does it preserve meaning?
```

**Decision matrix:**

- If quality is good (>75% meaning preserved) → Use it
- If quality is poor (<75%) → Skip this method

**Expected gain:** +2-4% macro-F1 (if quality is good)

---

### **1C: Paraphrasing with mT5 (Multilingual T5)**

mT5 supports 101 languages including Filipino:

```python
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

# Load mT5 (multilingual T5)
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")

def paraphrase_filipino(text):
    # mT5 can paraphrase Filipino text
    input_text = f"paraphrase: {text}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512)
    outputs = model.generate(inputs, max_length=512, num_return_sequences=3)

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Example:
text = "Ang presidente ay nangako ng pagbabago"
paraphrases = paraphrase_filipino(text)
```

**Expected gain:** +3-5% macro-F1

---

### **1D: Code-Switching Augmentation (Unique to Filipino)**

Filipino text often mixes English and Tagalog. Augment by varying the code-switching:

```python
import random

# Define English-Tagalog pairs
translations = {
    "government": "gobyerno",
    "help": "tulong",
    "people": "mga tao",
    "should": "dapat",
    "give": "magbigay",
    # ... add more pairs
}

def code_switch_augment(text):
    words = text.split()
    augmented = []

    for word in words:
        # Randomly switch between English and Tagalog
        if word in translations and random.random() < 0.3:
            augmented.append(translations[word])
        elif word in translations.values() and random.random() < 0.3:
            # Reverse lookup
            english = [k for k, v in translations.items() if v == word][0]
            augmented.append(english)
        else:
            augmented.append(word)

    return ' '.join(augmented)
```

**Expected gain:** +1-2% macro-F1

---

### **Phase 1 Expected Result: 73-76% Macro-F1**

---

## 📋 PHASE 2: MANUAL DATA COLLECTION (CRITICAL FOR >80%)

**Hard Truth:** To reach >80%, you need **high-quality, manually-labeled data**.

### **2A: Targeted Data Collection**

Focus on the weakest classes:

**Objective Class (90 samples → 300+ samples):**

1. Source: Filipino news articles (GMA, ABS-CBN, Rappler)

   - Filter for objective reporting
   - Avoid opinion pieces
   - Target: 100 new samples

2. Source: Government press releases

   - Official statements, statistics
   - Target: 50 new samples

3. Source: Academic abstracts (UP, ADMU)
   - Research papers in Filipino
   - Target: 50 new samples

**Neutral Class (401 samples → 700+ samples):**

1. Source: Customer service queries

   - "Paano po ba mag-apply?"
   - "Saan po yung office?"
   - Target: 150 new samples

2. Source: Product descriptions
   - Technical specifications
   - Target: 100 new samples

**Time Investment:**

- 8-12 hours of manual collection
- 4-6 hours of validation

**Expected gain:** +3-5% macro-F1 → **76-81% total**

---

### **2B: Crowdsourcing (Faster but Requires Budget)**

Platforms:

- Amazon MTurk (Filipino workers)
- Appen
- Scale AI

**Task:**

- Label 500-1000 political texts
- Focus on Objective and Neutral classes

**Cost:** ~$200-500 USD  
**Time:** 2-3 days  
**Expected gain:** +4-6% macro-F1 → **77-82% total**

---

## 📋 PHASE 3: ENSEMBLE METHODS

Combine mBERT + XLM-RoBERTa for better predictions:

```python
# Weighted ensemble
mbert_pred = mbert_model.predict(text)
xlmr_pred = xlmr_model.predict(text)

# Weighted average (tune weights on validation set)
final_pred = 0.4 * mbert_pred + 0.6 * xlmr_pred
```

**Expected gain:** +2-4% macro-F1 → **78-86% total**

---

## 📋 PHASE 4: ACTIVE LEARNING & REFINEMENT

1. **Identify misclassified samples** (especially near decision boundaries)
2. **Manually review and relabel** if needed
3. **Retrain** with corrected labels
4. **Iterate** 2-3 times

**Expected gain:** +1-3% macro-F1 → **79-89% total**

---

## 🚀 REALISTIC TIMELINE TO >80%

| Phase                              | Duration      | Cumulative F1 | Cost       |
| ---------------------------------- | ------------- | ------------- | ---------- |
| **Phase 1:** Filipino augmentation | 1-2 days      | 73-76%        | $0         |
| **Phase 2:** Manual collection     | 1-2 weeks     | 76-81%        | $0-500     |
| **Phase 3:** Ensemble              | 1 day         | 78-86%        | $0         |
| **Phase 4:** Active learning       | 1 week        | 79-89%        | $0         |
| **TOTAL**                          | **2-4 weeks** | **>80%** ✅   | **$0-500** |

---

## 💡 IMMEDIATE NEXT STEPS (START NOW)

### **Option A: Quick Wins (1-2 Days)**

**Goal:** Get to 73-76% quickly

1. Test back-translation quality with Filipino samples
2. If good → Use XLM-R contextual augmentation + back-translation
3. If bad → Use XLM-R contextual augmentation + mT5 paraphrasing only
4. Train Run #12

**Expected Result:** 73-76% macro-F1 (+5-8%)

---

### **Option B: Full Strategy (2-4 Weeks)**

**Goal:** Reach >80% reliably

1. **Week 1:** Filipino augmentation → 73-76%
2. **Week 2:** Manual data collection → 76-81%
3. **Week 3:** Ensemble methods → 78-86%
4. **Week 4:** Active learning → 79-89%

**Expected Result:** >80% macro-F1 ✅

---

## 🔧 UPDATED COLAB NOTEBOOK (FILIPINO-AWARE)

I need to create a **new notebook** with:

- XLM-RoBERTa contextual augmentation (Filipino-aware)
- mT5 paraphrasing (multilingual)
- Optional back-translation (with quality checks)
- **NO WordNet/English-only methods**

**Do you want me to create this updated notebook?**

---

## 📊 EXPECTED OUTCOMES

### **With Phase 1 Only (Filipino Augmentation):**

```
Current: 68.36% macro-F1
After Phase 1: 73-76% macro-F1 (+5-8%)
Gap to 80%: Still -4-7% short ⚠️
```

### **With Phases 1+2 (Augmentation + Manual Collection):**

```
Current: 68.36% macro-F1
After Phases 1+2: 76-81% macro-F1 (+8-13%)
Gap to 80%: 0-4% short ✅ CLOSE!
```

### **With All Phases (Complete Strategy):**

```
Current: 68.36% macro-F1
After All Phases: 79-89% macro-F1 (+11-21%)
Target >80%: ✅ ACHIEVABLE!
```

---

## 🚨 CRITICAL DECISION POINT

**You need to choose:**

### **Option 1: Fast Path (1-2 days)**

- Filipino-aware augmentation only
- Expected: 73-76% macro-F1
- ⚠️ Will NOT reach >80%

### **Option 2: Complete Path (2-4 weeks)**

- All 4 phases
- Expected: 79-89% macro-F1
- ✅ WILL reach >80%

### **Option 3: Hybrid (1 week)**

- Phase 1 (augmentation) + Phase 2 (manual collection)
- Expected: 76-81% macro-F1
- ⚠️ MIGHT reach >80% (not guaranteed)

---

## 📞 WHAT DO YOU WANT TO DO?

1. **Test back-translation quality first?** (I can create a test script)
2. **Create Filipino-aware augmentation notebook?** (XLM-R + mT5)
3. **Focus on manual data collection guide?** (Step-by-step)
4. **All of the above?** (Complete strategy)

**Which approach fits your timeline and target?** Let me know and I'll create the appropriate resources! 🚀
