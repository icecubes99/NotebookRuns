# 🔄 COLAB → LOCAL CONVERSION CHEAT SHEET

**Quick reference for adapting your Colab notebooks to run RemBERT locally**

---

## 📝 SECTION-BY-SECTION CHANGES

### **SECTION 1: Environment Setup**

#### ❌ **REMOVE (Colab-specific):**
```python
def pipi(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "--force-reinstall", "--no-cache-dir", *pkgs])

pipi(
    "numpy==2.1.1",
    "pandas==2.2.3",
    ...
)
```

#### ✅ **REPLACE WITH (Local):**
```python
# Assumes packages already installed via requirements_rembert.txt
import numpy as np
import pandas as pd
import torch
import transformers
from packaging import version

print("=== LOCAL ENVIRONMENT CHECK ===")
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

### **SECTION 3: Configuration**

#### 🔄 **UPDATE PATHS:**
```python
# ❌ Colab
CSV_PATH = '/content/augmented_adjudications_2025-10-22.csv'
OUT_DIR = "./runs_mbert_optimized"

# ✅ Local
CSV_PATH = 'd:/School/NotebookRuns/augmented_adjudications_2025-10-22.csv'
OUT_DIR = "./runs_rembert"
```

#### 🔄 **UPDATE MODEL CONFIG:**
```python
# ❌ mBERT
MODEL_CONFIGS = {
    "mbert": {"name": "bert-base-multilingual-cased", "desc": "mBERT (104 langs)"},
}
MODELS_TO_RUN = ["mbert"]

# ✅ RemBERT
MODEL_CONFIGS = {
    "rembert": {"name": "google/rembert", "desc": "RemBERT (110 langs, decoupled embeddings)"},
}
MODELS_TO_RUN = ["rembert"]
```

#### 🔄 **ADJUST HYPERPARAMETERS (RTX 3060):**
```python
# Recommended starting values
MAX_LENGTH = 256          # Balance efficiency vs context
BATCH_SIZE = 14           # Conservative for 12GB VRAM
GRAD_ACCUM_STEPS = 3      # Effective batch: 42
EPOCHS = 18               # Same as XLM-R best run
LR = 2.5e-5              # Between mBERT and XLM-R
WEIGHT_DECAY = 0.035     # Average
WARMUP_RATIO = 0.22      # Average
EARLY_STOP_PATIENCE = 7  # Average
```

---

### **SECTION 4: Data Loading**

#### ✅ **NO CHANGES NEEDED!**
- Data loading logic is path-agnostic
- Just ensure CSV_PATH points to correct local file
- Stratified splitting works the same

#### ⚠️ **VERIFY CSV EXISTS:**
```python
import os
assert os.path.exists(CSV_PATH), f"CSV not found: {CSV_PATH}"
print(f"✓ Dataset found: {CSV_PATH}")
print(f"  Size: {os.path.getsize(CSV_PATH) / 1024**2:.2f} MB")
```

---

### **SECTION 5-6: Dataset & Model**

#### ✅ **NO CHANGES NEEDED!**
- `TaglishDataset` works identically
- RemBERT uses same tokenizer API as mBERT
- `MultiTaskModel` architecture is model-agnostic
- Both use `token_type_ids` (unlike XLM-R)

---

### **SECTION 7-9: Training Loop**

#### ✅ **NO CHANGES NEEDED!**
- `compute_metrics_multi` is model-agnostic
- `MultiTaskTrainer` works with any AutoModel
- Focal loss, R-Drop, LLRD all compatible
- Class weighting logic unchanged

#### ⚠️ **DATALOADER WORKERS (Windows):**
```python
# In TrainingArguments, may need to adjust:
dataloader_num_workers = 0  # Windows can be flaky with multiprocessing
```

---

### **SECTION 10-12: Evaluation**

#### ✅ **NO CHANGES NEEDED!**
- Metrics calculation is model-agnostic
- Confusion matrices work the same
- Calibration code unchanged
- Just update output paths in save statements

---

## 🎯 CRITICAL CHANGES SUMMARY

### **Must Change:**
1. ✅ Model name: `"bert-base-multilingual-cased"` → `"google/rembert"`
2. ✅ CSV path: `/content/...` → `d:/School/NotebookRuns/...`
3. ✅ Output dir: `./runs_mbert_optimized` → `./runs_rembert`
4. ✅ Remove Section 1 auto-install (`pipi()` function)

### **Should Adjust:**
5. 🔧 `BATCH_SIZE`: 12 → 14 (optimize for RTX 3060)
6. 🔧 `MAX_LENGTH`: 320 → 256 (balance context vs speed)
7. 🔧 `MODELS_TO_RUN`: `["mbert"]` → `["rembert"]`

### **Optional Tweaks:**
8. 🎨 Update comments/print statements (change "mBERT" → "RemBERT")
9. 🎨 Adjust learning rate (try 2.5e-5 as starting point)
10. 🎨 Update run numbering (e.g., "Run #1" for RemBERT baseline)

---

## 🔍 VALIDATION CHECKLIST

Before running full training:

```python
# Quick validation script (paste at start of notebook)
import os
import torch
from transformers import AutoTokenizer, AutoModel

# 1. Check CUDA
assert torch.cuda.is_available(), "❌ CUDA not available!"
print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")

# 2. Check data
CSV_PATH = 'd:/School/NotebookRuns/augmented_adjudications_2025-10-22.csv'
assert os.path.exists(CSV_PATH), f"❌ CSV not found: {CSV_PATH}"
print(f"✅ Dataset found: {os.path.getsize(CSV_PATH) / 1024**2:.2f} MB")

# 3. Check model access
try:
    tokenizer = AutoTokenizer.from_pretrained("google/rembert")
    model = AutoModel.from_pretrained("google/rembert")
    print(f"✅ RemBERT loaded: {model.config.num_hidden_layers} layers, {model.config.hidden_size} hidden")
except Exception as e:
    print(f"❌ RemBERT load failed: {e}")

# 4. Check output directory
OUT_DIR = "./runs_rembert"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"✅ Output directory ready: {OUT_DIR}")

print("\n🚀 All checks passed! Ready to train.")
```

---

## 📊 EXPECTED DIFFERENCES

### **Performance:**
| Metric | mBERT (Colab) | RemBERT (Local, Expected) |
|--------|---------------|---------------------------|
| Training Time | 56-93 min | 50-75 min (RTX 3060 faster) |
| Macro-F1 | 63.06% | 65-70% (better architecture) |
| Objective F1 | ~40-50% | 45-55% (target improvement) |
| Neutral F1 | ~56-74% | 60-70% (target improvement) |

### **Resource Usage:**
| Resource | Colab T4 | RTX 3060 Local |
|----------|----------|----------------|
| VRAM | 16GB | 12GB (need optimization) |
| RAM | 12-15GB | Varies (8-16GB typical) |
| Speed | 1.0x | 1.2-1.5x (faster GPU) |
| Cost | Free (limited) | $0 (your hardware) |

---

## 🐛 QUICK FIXES

### **Path Issues:**
```python
# Use forward slashes (Python auto-converts)
CSV_PATH = 'd:/School/NotebookRuns/file.csv'  # ✅ Works on Windows

# OR raw strings
CSV_PATH = r'd:\School\NotebookRuns\file.csv'  # ✅ Also works

# AVOID mixing
CSV_PATH = 'd:\School/NotebookRuns\file.csv'  # ❌ Don't mix
```

### **VRAM Issues:**
```python
# If OOM errors occur:
BATCH_SIZE = 10              # Reduce from 14
MAX_LENGTH = 224             # Reduce from 256
GRAD_ACCUM_STEPS = 4         # Increase from 3 (maintains effective batch)
```

### **Slow First Run:**
```python
# First run downloads ~550MB model
# Subsequent runs use cache at:
# C:\Users\<YourName>\.cache\huggingface\transformers

# To specify custom cache:
import os
os.environ['HF_HOME'] = 'd:/School/NotebookRuns/hf_cache'
```

---

## 🎓 LEARNING NOTES

### **Why RemBERT May Outperform mBERT:**
1. **Decoupled embeddings:** Smaller input (128) + larger output (1152) = more efficient
2. **Better for multilingual:** Optimized specifically for low-resource languages
3. **Same training data:** Both trained on Wikipedia, but RemBERT uses smarter architecture
4. **Proven results:** Paper shows improvements over mBERT on classification tasks

### **When to Choose Each Model:**
- **mBERT:** Simple baseline, well-tested, 104 languages
- **XLM-R:** Best overall performance, 100 languages, RoBERTa architecture
- **RemBERT:** Balance efficiency/performance, 110 languages, Filipino-friendly

---

## 📞 TROUBLESHOOTING CONTACTS

If you encounter issues:

1. **Check guide:** `LOCAL_REMBERT_SETUP_GUIDE.md` (comprehensive troubleshooting)
2. **Verify setup:** Run `python -c "import torch; print(torch.cuda.is_available())"`
3. **Monitor GPU:** `nvidia-smi -l 1` in separate terminal
4. **Check logs:** Training errors usually show clear stack traces

---

**Quick Start:**
```powershell
# Setup (one-time)
.\setup_rembert_local.ps1

# Create notebook folder
mkdir [M]-REMBERT

# Copy and adapt mBERT notebook
# Update Sections 1, 3 as described above

# Run training
# Expected: 50-75 minutes for full run
```

**Good luck! 🚀**
