# 🖥️ LOCAL RemBERT TRAINING SETUP GUIDE

**Date:** October 27, 2025  
**Purpose:** Adapt Colab notebooks (mBERT/XLM-R) to run RemBERT training locally on Windows with RTX 3060  
**Target Model:** `google/rembert` (110 languages, optimized architecture)

---

## 📋 EXECUTIVE SUMMARY

### ✅ **Your Current Hardware**
- **GPU:** NVIDIA GeForce RTX 3060 (12GB VRAM) ✅ **SUFFICIENT**
- **CUDA:** Version 13.0 ✅ **SUPPORTED**
- **Python:** 3.13.5 ⚠️ **MAY NEED DOWNGRADE** (see compatibility section)
- **OS:** Windows (PowerShell)

### 🎯 **What Needs to Change**

| Aspect | Colab | Local (Windows) | Action Required |
|--------|-------|-----------------|-----------------|
| **Python Version** | 3.10.x | 3.13.5 | ⚠️ May need 3.10-3.12 for PyTorch compatibility |
| **Data Paths** | `/content/` | `d:\School\NotebookRuns\` | ✅ Simple path update |
| **Package Installation** | `pipi()` auto-install | Manual venv setup | 🔧 One-time setup |
| **GPU Detection** | Automatic | Needs CUDA PyTorch | ✅ Already have CUDA 13.0 |
| **Model Size** | RemBERT ~550MB | Same | ✅ Will download once |
| **Memory** | Colab: 12-15GB RAM | Your RAM varies | ⚠️ Need to check |
| **Output Directory** | `/content/runs_*` | Local folder | ✅ Simple path update |

---

## 🔍 RESEARCH FINDINGS

### 1️⃣ **RemBERT vs mBERT/XLM-R Comparison**

| Feature | mBERT | XLM-RoBERTa | RemBERT |
|---------|-------|-------------|---------|
| **Languages** | 104 | 100 | **110** ✅ |
| **Parameters** | 110M | 270M | **250M** |
| **Architecture** | BERT | RoBERTa | **BERT + Decoupled Embeddings** |
| **Input Embeddings** | 768 (tied) | 768 (tied) | **128 (small, decoupled)** ✅ |
| **Output Embeddings** | 768 (tied) | 768 (tied) | **1152 (large, discarded in fine-tuning)** ✅ |
| **Efficiency** | Baseline | 2.5x params | **More efficient than mBERT** ✅ |
| **Filipino Support** | Yes | Yes | **Yes** ✅ |
| **HuggingFace ID** | `bert-base-multilingual-cased` | `xlm-roberta-base` | `google/rembert` |

**Key Advantages of RemBERT:**
- ✅ **More efficient:** Smaller input embeddings save memory during training
- ✅ **More accurate:** Larger output embeddings improve representation (discarded after fine-tuning)
- ✅ **Better for low-resource languages:** Optimized for multilingual tasks
- ✅ **Same API:** Works with your existing code (Transformers library)

**Expected Performance:**
- **mBERT current:** 63.06% macro-F1 (Run #10)
- **XLM-R current:** 67.80% macro-F1 (Run #14)
- **RemBERT expected:** **65-70% macro-F1** (between mBERT and XLM-R, potentially better)

---

### 2️⃣ **Python Version Compatibility**

⚠️ **CRITICAL:** Python 3.13 may have issues with PyTorch/CUDA

| Python Version | PyTorch Support | Recommendation |
|----------------|-----------------|----------------|
| **3.13.x** | ⚠️ Limited/Experimental | Not recommended for production |
| **3.12.x** | ✅ Supported (PyTorch 2.2+) | **RECOMMENDED** |
| **3.11.x** | ✅ Fully supported | Good alternative |
| **3.10.x** | ✅ Fully supported | Most stable |

**Action:** Check if PyTorch 2.2+ works with Python 3.13, otherwise downgrade to 3.12.

---

### 3️⃣ **VRAM Requirements (RTX 3060 - 12GB)**

Based on your notebook configurations:

| Model | Batch Size | Grad Accum | Effective Batch | Sequence Length | Est. VRAM | Status |
|-------|-----------|------------|-----------------|-----------------|-----------|--------|
| **mBERT** | 12 | 4 | 48 | 320 | ~9-10GB | ✅ **FITS** |
| **XLM-R** | 20 | 3 | 60 | 224 | ~10-11GB | ✅ **FITS** |
| **RemBERT** | 16 | 3 | 48 | 256 | ~10-11GB | ✅ **SHOULD FIT** |

**Recommended RemBERT Settings for RTX 3060:**
```python
BATCH_SIZE = 12-16        # Start with 12, increase if stable
GRAD_ACCUM_STEPS = 3-4    # Effective batch: 36-64
MAX_LENGTH = 224-256      # Compromise between mBERT/XLM-R
USE_GRADIENT_CHECKPOINTING = True  # Saves ~20-30% VRAM
```

**Safety Margin:** Leave ~1-2GB for OS/driver overhead.

---

### 4️⃣ **Package Dependencies**

Your notebooks use these pinned versions (need to adapt for local):

```python
# Colab versions (from Section 1)
numpy==2.1.1
pandas==2.2.3
scikit-learn==1.5.2
matplotlib==3.9.2
transformers==4.44.2
accelerate==0.34.2
datasets==2.21.0
torch==2.2.2  # CUDA 11.8 or 12.1
```

**Local Installation Strategy:**
1. ✅ Use a **virtual environment** (venv or conda)
2. ✅ Install **PyTorch with CUDA 12.1** (matches your CUDA 13.0, backward compatible)
3. ✅ Pin **transformers==4.44.2** (tested version)
4. ✅ Install **packaging** (needed for version checks)

---

### 5️⃣ **File Path Changes**

All Colab notebooks use `/content/` paths. You need to replace with local paths:

| Colab Path | Local Path | Notes |
|------------|------------|-------|
| `/content/augmented_adjudications_2025-10-22.csv` | `d:/School/NotebookRuns/augmented_adjudications_2025-10-22.csv` | ✅ File exists (5.27 MB) |
| `/content/adjudications_2025-10-22.csv` | `d:/School/NotebookRuns/adjudications_2025-10-22.csv` | Check if exists |
| `./runs_mbert_optimized` | `./runs_rembert` | Output directory |
| `/content/calib_tmp` | `./runs_rembert/calib_tmp` | Temp calibration |

**Path Format for Windows in Python:**
```python
# Option 1: Forward slashes (Python handles conversion)
CSV_PATH = 'd:/School/NotebookRuns/augmented_adjudications_2025-10-22.csv'

# Option 2: Raw strings with backslashes
CSV_PATH = r'd:\School\NotebookRuns\augmented_adjudications_2025-10-22.csv'

# Option 3: os.path.join (most portable)
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'augmented_adjudications_2025-10-22.csv')
```

---

### 6️⃣ **Training Time Estimates**

Based on your Colab runs and RTX 3060 performance:

| Model | Colab Time | RTX 3060 Expected | Speed Factor |
|-------|-----------|-------------------|--------------|
| **mBERT** | 56-93 min | ~40-70 min | 1.2-1.5x faster (T4 vs 3060) |
| **XLM-R** | 75 min | ~55-65 min | 1.2-1.4x faster |
| **RemBERT** | N/A | **~50-75 min** | Estimate (between mBERT/XLM-R) |

**Factors:**
- ✅ RTX 3060 is faster than Colab T4 (1.2-1.5x)
- ⚠️ Local overhead (no preloaded cache) may slow first run
- ✅ Subsequent runs will be faster (cached models)

---

### 7️⃣ **Code Changes Required**

#### **Minimal Changes (Same Architecture):**
1. ✅ Change model name: `"bert-base-multilingual-cased"` → `"google/rembert"`
2. ✅ Update paths: `/content/` → `d:/School/NotebookRuns/`
3. ✅ Remove `pipi()` auto-install (Section 1)
4. ✅ Adjust `BATCH_SIZE` for RTX 3060 safety

#### **No Changes Needed:**
- ✅ Multi-task architecture (same API)
- ✅ Focal loss, R-Drop, LLRD (all compatible)
- ✅ Class weighting, oversampling (same logic)
- ✅ Tokenizer API (RemBERT uses WordPiece like mBERT)
- ✅ Calibration code (model-agnostic)

#### **Token Type IDs Check:**
```python
# RemBERT DOES use token_type_ids (like mBERT)
# No change needed in TaglishDataset class
self.use_token_type = "token_type_ids" in tokenizer.model_input_names
# Returns True for RemBERT (unlike XLM-R which returns False)
```

---

## 🚀 IMPLEMENTATION PLAN

### **Phase 1: Environment Setup** (15-30 minutes)

1. **Check/Install Python 3.12** (if 3.13 causes issues)
   ```powershell
   # Check current version
   python --version
   
   # If needed, download Python 3.12 from python.org
   # Install with "Add to PATH" option
   ```

2. **Create Virtual Environment**
   ```powershell
   # Navigate to project folder
   cd d:\School\NotebookRuns
   
   # Create venv
   python -m venv venv_rembert
   
   # Activate
   .\venv_rembert\Scripts\Activate.ps1
   
   # If execution policy error:
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Install PyTorch with CUDA**
   ```powershell
   # For CUDA 12.1 (compatible with CUDA 13.0)
   pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
   
   # Verify CUDA
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
   ```

4. **Install Dependencies**
   ```powershell
   pip install numpy==2.1.1 pandas==2.2.3 scikit-learn==1.5.2 matplotlib==3.9.2 seaborn
   pip install transformers==4.44.2 accelerate==0.34.2 datasets==2.21.0
   pip install packaging  # For version checks
   ```

5. **Verify Installation**
   ```powershell
   python -c "from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('google/rembert'); print('RemBERT tokenizer loaded successfully!')"
   ```

---

### **Phase 2: Create RemBERT Notebook** (30-60 minutes)

**Option A: Copy & Modify Existing Notebook**
```powershell
# Copy mBERT notebook as template
Copy-Item "[M]-MBERT\MBERT_TRAINING.ipynb" "[M]-REMBERT\REMBERT_TRAINING.ipynb"
```

**Option B: Create from Scratch** (recommended for learning)

Key changes in Section 3 (Configuration):
```python
# Model configuration
MODEL_CONFIGS = {
    "rembert": {"name": "google/rembert", "desc": "RemBERT (110 langs, decoupled embeddings)"},
}
MODELS_TO_RUN = ["rembert"]
OUT_DIR = "./runs_rembert"

# Paths (LOCAL)
CSV_PATH = 'd:/School/NotebookRuns/augmented_adjudications_2025-10-22.csv'

# Training config (RTX 3060 optimized)
MAX_LENGTH = 256          # Balanced (between mBERT 320 and XLM-R 224)
EPOCHS = 18               # Same as XLM-R Run #14
BATCH_SIZE = 14           # Conservative for 12GB VRAM
GRAD_ACCUM_STEPS = 3      # Effective batch: 42
LR = 2.5e-5              # Between mBERT and XLM-R
WEIGHT_DECAY = 0.035     # Average of both
WARMUP_RATIO = 0.22      # Average
EARLY_STOP_PATIENCE = 7  # Average
```

**Section 1 Replacement:**
```python
# ============================================================================
# SECTION 1: LOCAL ENVIRONMENT SETUP (Windows + RTX 3060)
# ============================================================================

import sys, os
import numpy as np
import pandas as pd
import torch
import transformers
from packaging import version

print("=== LOCAL ENVIRONMENT CHECK ===")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

print(f"\nCUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Verify transformers version
assert version.parse(transformers.__version__) >= version.parse("4.26.0"), \
    "Transformers too old for modern TrainingArguments."

print("\n✅ Environment check complete!")
```

---

### **Phase 3: Testing & Validation** (1-2 hours)

1. **Quick Smoke Test (1 epoch, small batch)**
   ```python
   # In Section 3, temporarily set:
   EPOCHS = 1
   BATCH_SIZE = 8
   GRAD_ACCUM_STEPS = 2
   ```

2. **Monitor VRAM Usage**
   ```powershell
   # In separate terminal, watch GPU usage
   nvidia-smi -l 1  # Update every 1 second
   ```

3. **Check Outputs**
   - [ ] Model loads successfully
   - [ ] Data loads from local path
   - [ ] Training starts without errors
   - [ ] VRAM stays under 11GB
   - [ ] Checkpoint saves to `./runs_rembert/`
   - [ ] Validation metrics appear
   - [ ] No ABI warnings (NumPy compatibility)

4. **Full Training Run**
   - Restore normal hyperparameters
   - Expected time: ~50-75 minutes
   - Monitor for OOM errors (reduce batch if needed)

---

### **Phase 4: Performance Comparison**

After training, compare with existing models:

| Metric | mBERT (Run #10) | XLM-R (Run #14) | RemBERT (Expected) | Status |
|--------|-----------------|------------------|-------------------|--------|
| **Overall Macro-F1** | 63.06% | 67.80% | 65-70% | 🎯 Target |
| **Sentiment F1** | ~63% | ~66% | 64-68% | - |
| **Polarization F1** | ~63% | ~62% | 62-67% | - |
| **Objective F1** | ~40-50% | ~41% | 45-55% | 🔑 Key metric |
| **Neutral F1** | ~56-74% | ~57% | 60-70% | 🔑 Key metric |
| **Training Time** | 56-93 min | 75 min | 50-75 min | - |

---

## 🔧 TROUBLESHOOTING GUIDE

### **Issue 1: PyTorch CUDA Not Available**
```powershell
# Symptoms
python -c "import torch; print(torch.cuda.is_available())"  # Returns False

# Solutions
# 1. Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# 2. Verify NVIDIA driver
nvidia-smi

# 3. Check CUDA toolkit (optional, PyTorch includes it)
```

---

### **Issue 2: Out of Memory (OOM) Error**
```python
# Symptoms
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB

# Solutions (try in order)
# 1. Reduce batch size
BATCH_SIZE = 10  # Down from 14

# 2. Reduce sequence length
MAX_LENGTH = 224  # Down from 256

# 3. Enable gradient checkpointing (should already be on)
USE_GRADIENT_CHECKPOINTING = True

# 4. Reduce gradient accumulation (but lowers effective batch)
GRAD_ACCUM_STEPS = 2

# 5. Use FP16 mixed precision (already enabled in TrainingArguments)
fp16 = True

# 6. Clear CUDA cache between runs
import torch
torch.cuda.empty_cache()
```

---

### **Issue 3: NumPy ABI Compatibility Warning**
```python
# Symptoms
RuntimeWarning: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x

# Solutions
# Downgrade NumPy (safest)
pip install numpy==1.26.4  # Last 1.x version

# OR use NumPy 2.x compatible builds (if available)
pip install --upgrade numpy pandas scikit-learn
```

---

### **Issue 4: Slow Data Loading**
```python
# Symptoms
Data loading takes minutes per epoch

# Solutions
# 1. Reduce dataloader workers (Windows issue)
# In TrainingArguments:
dataloader_num_workers = 0  # Single-threaded (slower but stable)

# 2. Pin memory (already in code)
dataloader_pin_memory = True

# 3. Cache dataset preprocessing
# (Already handled by TaglishDataset)
```

---

### **Issue 5: Model Download Fails**
```python
# Symptoms
ConnectionError / TimeoutError when loading RemBERT

# Solutions
# 1. Manual download (if internet is slow)
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("google/rembert", cache_dir="./rembert_cache")
model = AutoModel.from_pretrained("google/rembert", cache_dir="./rembert_cache")

# 2. Use HF_HOME environment variable
import os
os.environ['HF_HOME'] = 'd:/School/NotebookRuns/huggingface_cache'

# 3. Offline mode (after first download)
os.environ['TRANSFORMERS_OFFLINE'] = '1'
```

---

## 📊 EXPECTED OUTCOMES

### **Best Case Scenario** (Everything Optimal)
- ✅ Training completes in 50-60 minutes
- ✅ RemBERT achieves 68-70% macro-F1 (beats both mBERT and XLM-R)
- ✅ Objective class F1 improves to 50-55% (best so far)
- ✅ Neutral class F1 reaches 65-70%
- ✅ VRAM usage stays under 11GB
- ✅ No crashes or errors

### **Realistic Scenario** (Minor Issues)
- 🟡 Training takes 60-75 minutes (still faster than Colab)
- 🟡 RemBERT achieves 65-68% macro-F1 (between mBERT and XLM-R)
- 🟡 Need to reduce batch size to 10-12 for stability
- 🟡 Some VRAM fluctuations near 11-11.5GB
- 🟡 1-2 minor compatibility warnings (non-blocking)

### **Worst Case Scenario** (Significant Issues)
- ❌ Python 3.13 incompatibility requires downgrade to 3.12
- ❌ OOM errors require MAX_LENGTH reduction to 192
- ❌ NumPy 2.x issues require downgrade to 1.26.4
- ❌ Training takes 90+ minutes due to dataloader issues
- ❌ RemBERT underperforms (60-63% F1, same as mBERT)

---

## 🎯 SUCCESS CRITERIA

**Phase 1 (Setup) - MUST PASS:**
- [ ] Virtual environment created and activated
- [ ] PyTorch with CUDA 12.1 installed successfully
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] All dependencies installed without conflicts
- [ ] RemBERT tokenizer loads without errors

**Phase 2 (Testing) - MUST PASS:**
- [ ] 1-epoch smoke test completes without OOM
- [ ] VRAM usage under 11.5GB during training
- [ ] Data loads from local CSV path
- [ ] Checkpoints save to `./runs_rembert/`
- [ ] No critical errors in logs

**Phase 3 (Full Training) - TARGET:**
- [ ] Full training completes in under 90 minutes
- [ ] Overall macro-F1 ≥ 65% (beats mBERT baseline)
- [ ] Objective F1 ≥ 45% (improvement over current ~41%)
- [ ] Neutral F1 ≥ 60%
- [ ] Model artifacts saved correctly

**Phase 4 (Comparison) - BONUS:**
- [ ] RemBERT outperforms mBERT on at least 2/3 metrics
- [ ] Training time faster than Colab equivalent
- [ ] Reproducible across multiple runs (seed stability)
- [ ] Calibration improves test F1 by +1-2%

---

## 📝 NEXT STEPS AFTER SUCCESSFUL RUN

1. **Document Results**
   - Create `[M]-REMBERT/RUN_ANALYSIS.md`
   - Record all metrics in `run-data.md`
   - Compare with mBERT/XLM-R in summary table

2. **Iterate if Needed**
   - Adjust hyperparameters (LR, weight decay, class weights)
   - Try different MAX_LENGTH values (224, 256, 288, 320)
   - Test seed ensemble (seeds 42, 43, 44, 45)

3. **Ensemble Strategy**
   - Average predictions from mBERT + XLM-R + RemBERT
   - Expected boost: +1-3% macro-F1
   - Target: Push from 67-70% → 70-73%

4. **Data Augmentation**
   - Use RemBERT with augmented data (13,063 samples)
   - Expected boost: +2-5% macro-F1
   - Target: Push from 70-73% → 75%+ (BREAKTHROUGH!)

---

## 🔗 REFERENCES

1. **RemBERT Paper:** [Rethinking Embedding Coupling in Pre-trained Language Models](https://arxiv.org/abs/2010.12821)
2. **HuggingFace Model:** https://huggingface.co/google/rembert
3. **PyTorch CUDA Installation:** https://pytorch.org/get-started/locally/
4. **Transformers Documentation:** https://huggingface.co/docs/transformers/

---

## ✅ CHECKLIST

**Before Starting:**
- [ ] Read this entire guide
- [ ] Backup current notebooks
- [ ] Ensure 20GB+ free disk space
- [ ] Close unnecessary applications (free RAM/VRAM)
- [ ] Open Task Manager to monitor GPU usage

**During Setup:**
- [ ] Follow Phase 1 steps sequentially
- [ ] Verify each installation step
- [ ] Test CUDA availability before proceeding
- [ ] Create new folder `[M]-REMBERT/`

**During Training:**
- [ ] Monitor nvidia-smi in separate terminal
- [ ] Check training logs for warnings
- [ ] Save checkpoints every epoch
- [ ] Take notes on any errors

**After Training:**
- [ ] Verify output files exist
- [ ] Run calibration (Section 11A)
- [ ] Compare metrics with mBERT/XLM-R
- [ ] Document in run-data.md

---

**Good luck! 🚀 RemBERT has strong potential to improve on your current results, especially for multilingual Filipino text classification.**
