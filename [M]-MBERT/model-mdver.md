# SECTION 1

```python
# ============================================================================
# SECTION 1: ENVIRONMENT SETUP (ROBUST, PY3.12-FRIENDLY)
# ============================================================================

import sys, subprocess, importlib, os

def pipi(*pkgs):
    # Force reinstall + no cache to avoid stale wheels
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "--force-reinstall", "--no-cache-dir", *pkgs])

print("Installing pinned, compatible versions …")
# Torch: keep your existing CUDA build. If you don't have torch yet, uncomment the torch trio below.
# pipi("torch==2.2.2", "torchaudio==2.2.2", "torchvision==0.17.2")

# Pin NumPy 2.x and libs that are built against it
pipi(
    "numpy==2.1.1",
    "pandas==2.2.3",
    "scikit-learn==1.5.2",
    "matplotlib==3.9.2",
    "transformers==4.44.2",
    "accelerate==0.34.2",
    "datasets==2.21.0",
)

# --- Import order matters; import numpy FIRST to catch ABI issues clearly
import numpy as np
print("NumPy:", np.__version__)

# Now the rest
import torch, transformers, datasets, sklearn, pandas as pd, matplotlib, importlib

print("\n=== VERSION CHECK ===")
print("torch          :", getattr(torch, "__version__", "n/a"))
print("transformers   :", transformers.__version__)
print("accelerate     :", importlib.import_module("accelerate").__version__)
print("datasets       :", datasets.__version__)
print("scikit-learn   :", sklearn.__version__)
print("pandas         :", pd.__version__)
print("numpy          :", np.__version__)
print("matplotlib     :", matplotlib.__version__)

# Sanity for TrainingArguments modern kwargs
from packaging import version
assert version.parse(transformers.__version__) >= version.parse("4.26.0"), \
    "Transformers too old for `evaluation_strategy`."

# If NumPy was previously imported in this session, you may still have stale .so’s in memory.
# Simple guard: if you see an ABI error above, Restart runtime and run this cell again first.
print("\nCUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Count:", torch.cuda.device_count())
    print("Current CUDA Device:", torch.cuda.get_device_name(0))

```

# SECTION 1.5

```python
# ============================================================================
# SECTION 1.5: VERSION CHECK + TRAININGARGUMENTS COMPATIBILITY SHIM
# ============================================================================

import inspect, importlib, sys
import transformers as _tf

print("Transformers version loaded in memory:", _tf.__version__)

def _supported_kwargs_of_training_args():
    # Build the set of supported __init__ kwargs for the loaded TrainingArguments
    try:
        from transformers import TrainingArguments
        sig = inspect.signature(TrainingArguments.__init__)
        return set(sig.parameters.keys())
    except Exception as e:
        print("[Compat] Could not inspect TrainingArguments:", e)
        return set()

_SUPPORTED_TA_KEYS = _supported_kwargs_of_training_args()
print("Sample of supported TrainingArguments kwargs:", sorted(list(_SUPPORTED_TA_KEYS))[:12], "...")

def make_training_args_compat(**kwargs):
    """
    Create TrainingArguments while dropping any kwargs unsupported by the loaded transformers version.
    Prints what was ignored so you know if your runtime is old.
    """
    from transformers import TrainingArguments
    filtered = {k: v for k, v in kwargs.items() if k in _SUPPORTED_TA_KEYS}
    ignored = [k for k in kwargs.keys() if k not in _SUPPORTED_TA_KEYS]
    if ignored:
        print("[Compat] Ignored unsupported TrainingArguments keys:", ignored)
    return TrainingArguments(**filtered)

def get_early_stopping_callbacks(patience: int):
    """Return EarlyStoppingCallback if available; otherwise return []."""
    try:
        from transformers import EarlyStoppingCallback
        return [EarlyStoppingCallback(early_stopping_patience=patience)]
    except Exception as e:
        print("[Compat] EarlyStoppingCallback unavailable:", e)
        return []

```

# SECTION 2

```python

# ============================================================================
# SECTION 2: IMPORTS AND BASIC SETUP
# ============================================================================

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import time
from datetime import timedelta

# ============================================================================
# TIMING UTILITY - Track execution time for each section
# ============================================================================
class SectionTimer:
    def __init__(self):
        self.section_times = {}
        self.start_time = None
        self.total_start = time.time()

    def start_section(self, section_name):
        """Start timing a section"""
        self.start_time = time.time()
        print(f"\n🚀 Starting {section_name}...")

    def end_section(self, section_name):
        """End timing and display results"""
        if self.start_time is None:
            self.start_time = time.time()

        elapsed = time.time() - self.start_time
        self.section_times[section_name] = elapsed

        # Format time nicely
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        elif elapsed < 3600:
            time_str = f"{elapsed/60:.1f}m {elapsed%60:.0f}s"
        else:
            time_str = f"{elapsed/3600:.1f}h {(elapsed%3600)/60:.0f}m"

        total_elapsed = time.time() - self.total_start
        if total_elapsed < 60:
            total_str = f"{total_elapsed:.1f}s"
        elif total_elapsed < 3600:
            total_str = f"{total_elapsed/60:.1f}m {total_elapsed%60:.0f}s"
        else:
            total_str = f"{total_elapsed/3600:.1f}h {(total_elapsed%3600)/60:.0f}m"

        print(f"✅ {section_name} completed in {time_str}")
        print(f"🕒 Total runtime so far: {total_str}")
        print("-" * 60)

    def get_summary(self):
        """Get timing summary"""
        total = time.time() - self.total_start
        print("\n" + "="*60)
        print("⏱️  EXECUTION TIME SUMMARY")
        print("="*60)
        for section, elapsed in self.section_times.items():
            if elapsed < 60:
                time_str = f"{elapsed:.1f}s"
            elif elapsed < 3600:
                time_str = f"{elapsed/60:.1f}m {elapsed%60:.0f}s"
            else:
                time_str = f"{elapsed/3600:.1f}h {(elapsed%3600)/60:.0f}m"
            print(f"{section:<40} : {time_str}")

        if total < 60:
            total_str = f"{total:.1f}s"
        elif total < 3600:
            total_str = f"{total/60:.1f}m {total%60:.0f}s"
        else:
            total_str = f"{total/3600:.1f}h {(total%3600)/60:.0f}m"

        print(f"{'='*40} : {'='*10}")
        print(f"{'TOTAL EXECUTION TIME':<40} : {total_str}")
        print("="*60)

# Initialize global timer
timer = SectionTimer()
timer.start_section("SECTION 2: Environment & Imports")
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# End timing for section 2
timer.end_section("SECTION 2: Environment & Imports")
timer.start_section("SECTION 3: Configuration Setup")
```

```python
import os, random, json, math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer, AutoModel, TrainingArguments, Trainer,
    DataCollatorWithPadding, EarlyStoppingCallback, set_seed
)

def seed_all(seed=42):
    import transformers
    transformers.set_seed(seed)  # 🔧 RUN #9 FIX: Comprehensive transformers seed (sets all seeds including random, numpy, torch)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
```

# SECTION 3

```python
# 🤖 TRAINING ONLY: mBERT (bert-base-multilingual-cased)\n
# Expected: ~35-40 min, 60-65% macro-F1\n

# ===== Section 3 — Config (pooling + R-Drop + LLRD) =====

data_path = '/content/augmented_adjudications_2025-10-22.csv'
CSV_PATH = '/content/augmented_adjudications_2025-10-22.csv'


TITLE_COL = "Title"
TEXT_COL  = "Comment"
SENT_COL  = "Final Sentiment"
POL_COL   = "Final Polarization"

MODEL_CONFIGS = {
    "mbert": {"name": "bert-base-multilingual-cased", "desc": "mBERT (104 langs)"},
}
MODELS_TO_RUN = ["mbert"]  # ← TRAINING ONLY mBERT
OUT_DIR = "./runs_mbert_optimized"  # ← Separate output directory

# ============================================================================
# CORE TRAINING - RUN #10-13: SEED ENSEMBLE STRATEGY (4 SEEDS FOR +1-2% BOOST)
# Run #9 Result: 62.74% Macro-F1 (✅ NEW BEST! +0.68% over R4!)
# Key Finding: Seed matters! Same hyperparameters, different seed → +0.68% improvement
# Run #10-13 Strategy: TRAIN 4 MODELS WITH DIFFERENT SEEDS (42, 43, 44, 45) + ENSEMBLE
# Expected Outcome:
#   - Each model: 62-63% Macro-F1 (consistent with R9)
#   - Ensemble (average predictions): 64-65% Macro-F1 (+1-2% from ensemble diversity)
# How to use:
#   Run #10: Set SEED = 42 (already done in R9, baseline)
#   Run #11: Set SEED = 43, re-run notebook
#   Run #12: Set SEED = 44, re-run notebook
#   Run #13: Set SEED = 45, re-run notebook
#   Then ensemble predictions by averaging logits/probabilities
# Expected training time per model: ~93 minutes (same as R9)
# Total time for 4 models: ~6 hours
# ============================================================================

# 🎯 SEED CONFIGURATION
# Run #10-13: Seed ensemble complete (58.28% to 63.06% variance - seed 42 is best!)
# Run #14: ARCHITECTURAL DISASTER (-5.3%) - proved massive changes are harmful!
# Run #15: SEQUENCE LENGTH OPTIMIZATION - Research-backed quick win!
SEED = 42  # 🔥 Use best seed (42) proven in R9-R10

# ============================================================================
# CORE HYPERPARAMETERS - RUN #15: SEQUENCE LENGTH OPTIMIZATION
# Strategy: R9 proven config + RESEARCH-BACKED sequence length increase
# Research: Sequence length 320 achieved 71.7% F1 in NIH study (PMC9051848)
# Expected: +1-3% from better context capture
# ============================================================================
MAX_LENGTH = 320            # ⬆️ RESEARCH-BACKED OPTIMAL (was 224, +43% more context!)
EPOCHS = 20                 # ✅ RESTORE R9 PROVEN (R14's changes failed!)
BATCH_SIZE = 12            # ⬇️ KEEP 12 (compensate for longer sequences)
LR = 2.5e-5               # ✅ RESTORE R9 PROVEN (2.0e-5 in R14 was too conservative)
WEIGHT_DECAY = 0.03       # ✅ RESTORE R9 PROVEN (R14's 0.04 was over-regularizing)
WARMUP_RATIO = 0.20       # ✅ RESTORE R9 PROVEN (R14's 0.25 was unnecessary)
EARLY_STOP_PATIENCE = 8   # ✅ RESTORE R9 PROVEN (R14's 10 allowed overfitting)
GRAD_ACCUM_STEPS = 4      # ⬆️ KEEP 4 (maintain effective batch 48 for longer sequences)

# Per-task loss - RUN #10-13: R9 CONFIGURATION (PROVEN OPTIMAL)
USE_FOCAL_SENTIMENT = True
USE_FOCAL_POLARITY  = True
FOCAL_GAMMA_SENTIMENT = 2.5   # ✅ R9 PROVEN
FOCAL_GAMMA_POLARITY = 3.5    # ✅ R9 PROVEN
LABEL_SMOOTH_SENTIMENT = 0.10 # ✅ R9 PROVEN
LABEL_SMOOTH_POLARITY = 0.08  # ✅ R9 PROVEN

# Task weights - RUN #10-13: R9 CONFIGURATION (PROVEN OPTIMAL)
TASK_LOSS_WEIGHTS = {"sentiment": 1.0, "polarization": 1.4}  # ✅ R9 PROVEN

# Gradient control - RUN #10-13: R9 CONFIGURATION (task-specific disabled, proven harmful in R7!)
USE_TASK_SPECIFIC_GRAD_NORM = False  # ✅ DISABLED (R7 proved this is HARMFUL!)
MAX_GRAD_NORM = 0.5          # ✅ R9 PROVEN (from R4)
USE_GRADIENT_CHECKPOINTING = True  # Memory efficiency

# ============================================================================
# AGGRESSIVE CLASS WEIGHTS - RUN #10-13: R9 CONFIGURATION (PROVEN OPTIMAL)
# Run #5 showed: 1.30 negative weight WORSENED recall (47.5%→40.3%)!
# negative: Keep R9's 1.10 (1.30 backfired spectacularly)
# neutral: Keep R9's 1.80 (working with oversampling)
# objective: Keep R4's 2.50 (working well)
# ============================================================================
CLASS_WEIGHT_MULT = {
    "sentiment": {
        "negative": 1.10,    # ✅ RESTORE R4 from 1.30 (1.30 made recall WORSE!)
        "neutral":  1.80,    # ✅ RESTORE R4 (working with oversampling)
        "positive": 1.30     # ✅ RESTORE R4
    },
    "polarization": {
        "non_polarized": 1.20,  # ✅ RESTORE R4
        "objective":     2.50,  # ✅ RESTORE R4 (working well)
        "partisan":      0.95   # ✅ RESTORE R4
    }
}

# Cap maximum class weight to prevent instability
MAX_CLASS_WEIGHT = 10.0  # 🔥 INCREASED (was 6.0) - Allow stronger weighting for weak classes

# ============================================================================
# SELECTIVE OVERSAMPLING - RUN #10-13: R9 CONFIGURATION (PROVEN OPTIMAL)
# Run #5 showed: 10x objective + 3x neutral = DISASTER (non-polarized -8.2%!)
# Run #4's 8.5x/3.5x balance was OPTIMAL - proven stable in R9
# Goal: Keep R9 configuration, only vary seed for ensemble
# ============================================================================
USE_OVERSAMPLING = True
USE_JOINT_OVERSAMPLING = True
USE_SMART_OVERSAMPLING = True
JOINT_ALPHA = 0.70              # ✅ RESTORE R4 (was effective)
JOINT_OVERSAMPLING_MAX_MULT = 8.0  # ✅ RESTORE R4
OBJECTIVE_BOOST_MULT = 8.5      # ✅ RESTORE R4 from 10.0 (10x destroyed non-polarized!)
NEUTRAL_BOOST_MULT = 3.5        # ✅ RESTORE R4 from 3.0 (3x removed critical signal!)

# ============================================================================
# ARCHITECTURE - RUN #15: RESTORE R9 PROVEN SIMPLE DESIGN
# R14 Lesson: Massive architectural changes (-5.3% regression!) proved harmful!
# R15 Strategy: Restore proven simple architecture, focus on data/sequence length
# Goal: +1-3% improvement from sequence length optimization (Target: 64-66%)
# ============================================================================
HEAD_HIDDEN = 768            # ✅ RESTORE R9 PROVEN (R14's 1536 caused severe overfitting!)
HEAD_DROPOUT = 0.25          # ✅ RESTORE R9 PROVEN (R14's 0.30 was over-regularizing)
REP_POOLING = "last4_mean"   # ✅ RESTORE R9 PROVEN (R14's attention pooling had issues)
HEAD_LAYERS = 3              # ✅ RESTORE R9 PROVEN (R14's 4 layers were too deep)

# ❌ REMOVE ALL R14 ARCHITECTURAL COMPLEXITY (ALL PROVEN HARMFUL!)
USE_TASK_SPECIFIC_PROJECTIONS = False   # ❌ DISABLED (reduced shared learning)
USE_RESIDUAL_CONNECTIONS = False        # ❌ DISABLED (over-engineering, no benefit)
USE_MULTI_SAMPLE_DROPOUT = False        # ❌ DISABLED (increased time, unstable)
MULTI_SAMPLE_DROPOUT_NUM = 1            # ❌ NOT USED (disabled)
USE_ATTENTION_POOLING_MULTI_HEAD = False # ❌ DISABLED (caused float16 issues)

# ============================================================================
# REGULARIZATION - RUN #10-13: R9 CONFIGURATION (PROVEN OPTIMAL)
# ============================================================================
USE_RDROP = True
RDROP_ALPHA = 0.6           # ✅ R9 PROVEN (from R4)
RDROP_WARMUP_EPOCHS = 2     # ✅ R9 PROVEN (from R4)

# LLRD (layer-wise learning-rate decay) - RUN #10-13: R9 CONFIGURATION
USE_LLRD = True
LLRD_DECAY = 0.90            # ✅ R9 PROVEN (from R4)
HEAD_LR_MULT = 3.0           # ✅ R9 PROVEN (from R4)

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# RUN #16 OVERRIDES: modest head capacity bump (+16.7%)
# Keep architecture simple; change only head size and dropout.
HEAD_HIDDEN = 896
HEAD_DROPOUT = 0.28
# ---------------------------------------------------------------------------

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================
print("="*80)
print(f"🎯 mBERT TRAINING - RUN #15: SEQUENCE LENGTH OPTIMIZATION (SEED = {SEED})")
print("="*80)
print(f"📊 Run History:")
print(f"   R1→R8: 58.5%-61.99% (hyperparameter tuning)")
print(f"   R9-R10: 62.74%-63.06% 🏆 BEST BASELINE (seed 42)")
print(f"   R11-R13: 58.28%-62.45% (seed ensemble - 4.78% variance)")
print(f"   R14: 57.76% ❌ DISASTER (-5.3%) - architectural overreach failed!")
print(f"   Key Finding: Simple architecture works best, need data-centric approach!")
print(f"   R15 Strategy: SEQUENCE LENGTH OPTIMIZATION + PREPARE FOR AUGMENTATION")
print(f"")
print(f"🎲 Current Seed: {SEED} (best seed from R9-R13 experiments)")
print(f"")
print(f"🎯 Run #15 Goal: QUICK WIN VIA SEQUENCE LENGTH + DATA PREP")
print(f"   ├─ Current best: 63.06% (R10, seed 42)")
print(f"   ├─ Target: 64-66% Macro-F1 (+1-3% from sequence length)")
print(f"   ├─ Strategy: MAX_LENGTH 224 → 320 (research-backed optimal)")
print(f"   ├─ Next steps: Data augmentation for Track 1 (75% breakthrough)")
print(f"   └─ Long-term target: 75% Macro-F1 via multi-track strategy")
print(f"")
print(f"📏 SEQUENCE LENGTH OPTIMIZATION (RESEARCH-BACKED):")
print(f"   ├─ MAX_LENGTH: 224 → {MAX_LENGTH} (+43% more context!)")
print(f"   ├─ Research: Length 320 achieved 71.7% F1 in NIH study (PMC9051848)")
print(f"   ├─ Rationale: News articles have important late-text context")
print(f"   └─ Why it works: Sentiment/polarization signals may appear late in text")
print()
print(f"⏱️  Training Configuration (R9 PROVEN + SEQUENCE LENGTH ADJUSTMENT):")
print(f"   ├─ Epochs: {EPOCHS} ✅ R9 PROVEN")
print(f"   ├─ Batch Size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM_STEPS}) ⬇️ Adjusted for longer sequences")
print(f"   ├─ Learning Rate: {LR} ✅ R9 PROVEN")
print(f"   ├─ Weight Decay: {WEIGHT_DECAY} ✅ R9 PROVEN")
print(f"   ├─ Warmup Ratio: {WARMUP_RATIO} ✅ R9 PROVEN")
print(f"   ├─ Early Stop Patience: {EARLY_STOP_PATIENCE} ✅ R9 PROVEN")
print(f"   └─ Estimated Time: ~95-105 minutes (similar to R9, slightly longer sequences)")
print()
print(f"🏗️  Architecture (R9 PROVEN SIMPLE DESIGN - R14 COMPLEXITY REMOVED):")
print(f"   ├─ Head Hidden: {HEAD_HIDDEN} ✅ R9 PROVEN")
print(f"   ├─ Head Layers: {HEAD_LAYERS} ✅ R9 PROVEN")
print(f"   ├─ Head Dropout: {HEAD_DROPOUT} ✅ R9 PROVEN")
print(f"   ├─ Pooling: {REP_POOLING} ✅ R9 PROVEN")
print(f"   ├─ Task Projections: {USE_TASK_SPECIFIC_PROJECTIONS} ❌ DISABLED (R14 proved harmful)")
print(f"   ├─ Residual Connections: {USE_RESIDUAL_CONNECTIONS} ❌ DISABLED (R14 proved harmful)")
print(f"   └─ Multi-Sample Dropout: {USE_MULTI_SAMPLE_DROPOUT} ❌ DISABLED (R14 proved harmful)")
print()
print(f"⚖️  Critical Class Weights (R9 PROVEN):")
print(f"   ├─ Negative:  {CLASS_WEIGHT_MULT['sentiment']['negative']}x ✅ R9 PROVEN")
print(f"   ├─ Neutral:   {CLASS_WEIGHT_MULT['sentiment']['neutral']}x ✅ R9 PROVEN")
print(f"   ├─ Objective: {CLASS_WEIGHT_MULT['polarization']['objective']}x ✅ R9 PROVEN")
print(f"   └─ Max Weight Cap: {MAX_CLASS_WEIGHT} ✅ R9 PROVEN")
print()
print(f"📊 Oversampling Strategy (R9 PROVEN):")
print(f"   ├─ Joint Alpha: {JOINT_ALPHA} ✅ R9 PROVEN")
print(f"   ├─ Max Multiplier: {JOINT_OVERSAMPLING_MAX_MULT}x ✅ R9 PROVEN")
print(f"   ├─ Objective Boost: {OBJECTIVE_BOOST_MULT}x ✅ R9 PROVEN")
print(f"   └─ Neutral Boost: {NEUTRAL_BOOST_MULT}x ✅ R9 PROVEN")
print()
print(f"❌ TASK-SPECIFIC GRADIENT CONTROL: DISABLED (R7 proved this is HARMFUL!):")
print(f"   ├─ Feature Enabled: {USE_TASK_SPECIFIC_GRAD_NORM} ✅ DISABLED")
print(f"   ├─ Global MAX_GRAD_NORM: {MAX_GRAD_NORM} ✅ R9 PROVEN")
print(f"   └─ R7 Lesson: Task-specific gradients caused -7.91% regression (worst run ever)")
print()
print(f"🔥 Advanced Techniques (R9 PROVEN - ALL PARAMETERS FROM BEST RUN):")
print(f"   ├─ Focal Loss Gamma (Sent/Pol): {FOCAL_GAMMA_SENTIMENT}/{FOCAL_GAMMA_POLARITY} ✅ R9 PROVEN")
print(f"   ├─ Label Smoothing (Sent/Pol): {LABEL_SMOOTH_SENTIMENT}/{LABEL_SMOOTH_POLARITY} ✅ R9 PROVEN")
print(f"   ├─ Task Weights (Sent/Pol): {TASK_LOSS_WEIGHTS['sentiment']}/{TASK_LOSS_WEIGHTS['polarization']} ✅ R9 PROVEN")
print(f"   ├─ R-Drop Alpha: {RDROP_ALPHA} ✅ R9 PROVEN")
print(f"   └─ LLRD Decay: {LLRD_DECAY} ✅ R9 PROVEN")
print()
print(f"🎯 Run #15 Expected Results:")
print(f"   ├─ Target: 64-66% Macro-F1 (+1-3% from sequence length)")
print(f"   ├─ Strategy: Research-backed MAX_LENGTH=320 for better context")
print(f"   ├─ Next Step: Data augmentation (back-translation for weak classes)")
print(f"   └─ Breakthrough Path: Multi-track strategy to 75% (see BREAKTHROUGH_STRATEGY_TO_75_PERCENT.md)")
print()
print(f"📁 Output: {OUT_DIR}")
print("="*80)

# 🎲 Apply the configured seed (overrides the default seed_all(42) from Cell 6)
seed_all(SEED)
print(f"✅ Seed {SEED} applied successfully!")
print()

# End timing for section 3
timer.end_section("SECTION 3: Configuration Setup")
timer.start_section("SECTION 4: Data Loading & Preprocessing")
```

# SECTION 4

```python
# ===== Section 4 — Load & Prepare Data (updated for multipliers) =====
## Augmented-training aware defaults (local to this cell)
USE_AUGMENTED_TRAIN = globals().get('USE_AUGMENTED_TRAIN', True)
AUG_CSV_PATH = globals().get('AUG_CSV_PATH', '/content/augmented_adjudications_2025-10-22.csv')
if USE_AUGMENTED_TRAIN:
    # Soften manual boosts now that minority classes are larger
    CLASS_WEIGHT_MULT = {
        "sentiment": {"negative": 1.00, "neutral": 1.00, "positive": 1.20},
        "polarization": {"non_polarized": 1.00, "objective": 1.60, "partisan": 1.00},
    }
    MAX_CLASS_WEIGHT = 8.0
    USE_SMART_OVERSAMPLING = False
    JOINT_ALPHA = 0.30
    JOINT_OVERSAMPLING_MAX_MULT = 4.0
    OBJECTIVE_BOOST_MULT = 1.0
    NEUTRAL_BOOST_MULT = 1.0

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

required = [TITLE_COL, TEXT_COL, SENT_COL, POL_COL]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}. Found: {list(df.columns)}")

df = df.dropna(subset=[TITLE_COL, TEXT_COL, SENT_COL, POL_COL]).reset_index(drop=True)

# Encode labels
from sklearn.preprocessing import LabelEncoder
sent_le = LabelEncoder().fit(df[SENT_COL])
pol_le  = LabelEncoder().fit(df[POL_COL])

df["sent_y"] = sent_le.transform(df[SENT_COL])
df["pol_y"]  = pol_le.transform(df[POL_COL])

num_sent_classes = len(sent_le.classes_)
num_pol_classes  = len(pol_le.classes_)

print("Sentiment classes:", dict(enumerate(sent_le.classes_)))
print("Polarization classes:", dict(enumerate(pol_le.classes_)))

# Splits: joint stratify by both tasks (preserve distribution across sentiment x polarization)
from sklearn.model_selection import train_test_split
X = df[[TITLE_COL, TEXT_COL]].copy()
y_sent = df["sent_y"].values
y_pol  = df["pol_y"].values

y_joint = y_sent * 10 + y_pol
X_train, X_tmp, ysent_train, ysent_tmp, ypol_train, ypol_tmp = train_test_split(
    X, y_sent, y_pol, test_size=0.3, random_state=SEED, stratify=y_joint
)
joint_tmp = ysent_tmp * 10 + ypol_tmp
X_val, X_test, ysent_val, ysent_test, ypol_val, ypol_test = train_test_split(
    X_tmp, ysent_tmp, ypol_tmp, test_size=0.5, random_state=SEED, stratify=joint_tmp
)

# Append augmented rows to TRAIN only (no leakage into val/test)
if USE_AUGMENTED_TRAIN:
    if os.path.isfile(AUG_CSV_PATH):
        aug = pd.read_csv(AUG_CSV_PATH).dropna(subset=[TITLE_COL, TEXT_COL, SENT_COL, POL_COL])
        # Use same encoders to align indices
        aug["sent_y"] = sent_le.transform(aug[SENT_COL])
        aug["pol_y"]  = pol_le.transform(aug[POL_COL])
        X_train = pd.concat([X_train, aug[[TITLE_COL, TEXT_COL]]], ignore_index=True)
        ysent_train = np.concatenate([ysent_train, aug["sent_y"].values])
        ypol_train  = np.concatenate([ypol_train,  aug["pol_y"].values])
    else:
        print(f"[Warn] USE_AUGMENTED_TRAIN=True but file not found: {AUG_CSV_PATH}")

print("Train size:", len(X_train), "Val size:", len(X_val), "Test size:", len(X_test))

# Balanced class weights from TRAIN only
from sklearn.utils.class_weight import compute_class_weight
import numpy as np, json, os

def safe_class_weights(y, n_classes):
    classes = np.arange(n_classes)
    counts = np.bincount(y, minlength=n_classes)
    if np.any(counts == 0):
        return np.ones(n_classes, dtype=np.float32)
    return compute_class_weight("balanced", classes=classes, y=y).astype(np.float32)

sent_weights_np = safe_class_weights(ysent_train, num_sent_classes)
pol_weights_np  = safe_class_weights(ypol_train,  num_pol_classes)

# Apply user multipliers by class name
sent_name_to_idx = {name: i for i, name in enumerate(sent_le.classes_)}
pol_name_to_idx  = {name: i for i, name in enumerate(pol_le.classes_)}

for cname, mult in CLASS_WEIGHT_MULT["sentiment"].items():
    if cname in sent_name_to_idx:
        sent_weights_np[sent_name_to_idx[cname]] *= float(mult)

for cname, mult in CLASS_WEIGHT_MULT["polarization"].items():
    if cname in pol_name_to_idx:
        pol_weights_np[pol_name_to_idx[cname]] *= float(mult)

# Apply class weight caps to prevent training instability
sent_weights_np = np.clip(sent_weights_np, 0.1, MAX_CLASS_WEIGHT)
pol_weights_np = np.clip(pol_weights_np, 0.1, MAX_CLASS_WEIGHT)

print("Final sentiment class weights (capped):", {sent_le.classes_[i]: float(w) for i, w in enumerate(sent_weights_np)})
print("Final polarization class weights (capped):", {pol_le.classes_[i]: float(w) for i, w in enumerate(pol_weights_np)})
print(f"Class weights were capped at maximum: {MAX_CLASS_WEIGHT}")

# Save label maps
with open(os.path.join(OUT_DIR, "label_map_sentiment.json"), "w") as f:
    json.dump({int(k): v for k, v in dict(enumerate(sent_le.classes_)).items()}, f, indent=2)
with open(os.path.join(OUT_DIR, "label_map_polarization.json"), "w") as f:
    json.dump({int(k): v for k, v in dict(enumerate(pol_le.classes_)).items()}, f, indent=2)

# End timing for section 4
timer.end_section("SECTION 4: Data Loading & Preprocessing")
timer.start_section("SECTION 5-9: Model Architecture & Training Setup")

```

# SECTION 5 

```python
# ===== Section 5 — Dataset & Collator (proper text-pair encoding) =====
from torch.utils.data import Dataset

class TaglishDataset(Dataset):
    def __init__(self, titles, texts, y_sent, y_pol, tokenizer, max_length=224):
        self.titles = list(titles)
        self.texts  = list(texts)
        self.y_sent = np.array(y_sent)
        self.y_pol  = np.array(y_pol)
        self.tok = tokenizer
        self.max_length = max_length
        # mBERT has token_type_ids; XLM-R/RemBERT don't, and that's fine.
        self.use_token_type = "token_type_ids" in tokenizer.model_input_names

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Pass title as text, comment as text_pair so the tokenizer inserts the correct separators.
        # We also bias truncation to the comment since titles are short.
        enc = self.tok(
            text=str(self.titles[idx]),
            text_pair=str(self.texts[idx]),
            truncation="only_second",     # keep the title intact; trim the comment if needed
            max_length=self.max_length,
            return_token_type_ids=self.use_token_type,
        )
        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "sentiment_labels": torch.tensor(self.y_sent[idx], dtype=torch.long),
            "polarization_labels": torch.tensor(self.y_pol[idx], dtype=torch.long),
        }
        if self.use_token_type and "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"]
        return item

```

# SECTION 6

```python
# ===== Section 6 — Multi-Task Model (ENHANCED ARCHITECTURE FOR 75% TARGET) =====
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import math

def mean_pooling(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = (token_embeddings * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom

class AttentionPooling(nn.Module):
    """
    Learned attention pooling over layers and tokens.
    Dynamically learns which layers and tokens are most important.
    """
    def __init__(self, hidden_size, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        # Layer attention: learn which layers matter most
        self.layer_attention = nn.Linear(hidden_size, 1)
        # Token attention: learn which tokens matter most
        self.token_attention = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: tuple of (num_layers, batch_size, seq_len, hidden_size)
        attention_mask: (batch_size, seq_len)
        """
        # Stack last N layers: (num_layers, batch, seq, hidden)
        last_n = torch.stack(hidden_states[-self.num_layers:])

        # Layer-wise attention: which layers are most important?
        # Average across seq_len for each layer
        layer_repr = last_n.mean(dim=2)  # (num_layers, batch, hidden)
        layer_scores = self.layer_attention(layer_repr).squeeze(-1)  # (num_layers, batch)
        layer_weights = F.softmax(layer_scores, dim=0)  # (num_layers, batch)

        # Weighted combination of layers
        layer_weights = layer_weights.unsqueeze(2).unsqueeze(3)  # (num_layers, batch, 1, 1)
        weighted_hidden = (last_n * layer_weights).sum(dim=0)  # (batch, seq, hidden)

        # Token-wise attention: which tokens are most important?
        token_scores = self.token_attention(weighted_hidden).squeeze(-1)  # (batch, seq)

        # Mask padding tokens (use float16-safe value)
        # -1e4 is safe for both float32 and float16 (fp16 max ~65k)
        mask = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
        token_scores = token_scores.masked_fill(attention_mask == 0, -1e4)
        token_weights = F.softmax(token_scores, dim=1).unsqueeze(-1)  # (batch, seq, 1)

        # Weighted combination of tokens
        pooled = (weighted_hidden * token_weights).sum(dim=1)  # (batch, hidden)

        return pooled

class ResidualBlock(nn.Module):
    """
    Residual block with Layer Normalization and GELU activation.
    Helps with gradient flow in deeper networks.
    """
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # Xavier initialization for better gradient flow
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        # First sub-layer with residual
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x + residual

        # Second sub-layer with residual
        residual = x
        x = self.norm2(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x + residual

        return x

class MultiTaskModel(nn.Module):
    def __init__(self, base_model_name: str, num_sent: int, num_pol: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.hidden = self.encoder.config.hidden_size

        # 🔥 NEW: Attention-based pooling (learns what's important!)
        if REP_POOLING == "attention":
            self.attention_pool = AttentionPooling(self.hidden, num_layers=4)

        # 🔥 ENHANCED: Larger trunk with residual connections
        # Project from encoder hidden size to HEAD_HIDDEN
        self.trunk_proj = nn.Linear(self.hidden, HEAD_HIDDEN)

        # Add residual blocks for better gradient flow
        self.trunk_blocks = nn.ModuleList([
            ResidualBlock(HEAD_HIDDEN, dropout=HEAD_DROPOUT)
            for _ in range(2)  # 2 residual blocks
        ])

        self.trunk_norm = nn.LayerNorm(HEAD_HIDDEN)
        self.trunk_dropout = nn.Dropout(HEAD_DROPOUT)

        # Xavier initialization for projection
        nn.init.xavier_uniform_(self.trunk_proj.weight)
        nn.init.zeros_(self.trunk_proj.bias)

        # 🔥 NEW: Task-specific projections (learn task-specific features)
        self.sent_proj = nn.Sequential(
            nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(HEAD_HIDDEN),
            nn.Dropout(HEAD_DROPOUT * 0.5),
        )
        self.pol_proj = nn.Sequential(
            nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(HEAD_HIDDEN),
            nn.Dropout(HEAD_DROPOUT * 0.5),
        )

        # Initialize task projections
        for module in [self.sent_proj, self.pol_proj]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        # 🔥 ENHANCED: Deeper multi-layer heads with residuals
        if HEAD_LAYERS >= 4:
            # 4-layer heads with residual connections
            self.head_sent = nn.ModuleList([
                nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN),
                nn.GELU(),
                nn.LayerNorm(HEAD_HIDDEN),
                nn.Dropout(HEAD_DROPOUT * 0.8),
                ResidualBlock(HEAD_HIDDEN, dropout=HEAD_DROPOUT * 0.8),
                nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN // 2),
                nn.GELU(),
                nn.LayerNorm(HEAD_HIDDEN // 2),
                nn.Dropout(HEAD_DROPOUT * 0.7),
                nn.Linear(HEAD_HIDDEN // 2, num_sent)
            ])
            self.head_pol = nn.ModuleList([
                nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN),
                nn.GELU(),
                nn.LayerNorm(HEAD_HIDDEN),
                nn.Dropout(HEAD_DROPOUT * 0.8),
                ResidualBlock(HEAD_HIDDEN, dropout=HEAD_DROPOUT * 0.8),
                nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN // 2),
                nn.GELU(),
                nn.LayerNorm(HEAD_HIDDEN // 2),
                nn.Dropout(HEAD_DROPOUT * 0.7),
                nn.Linear(HEAD_HIDDEN // 2, num_pol)
            ])

            # Initialize head layers
            for head in [self.head_sent, self.head_pol]:
                for module in head:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        nn.init.zeros_(module.bias)

        elif HEAD_LAYERS == 3:
            # 3-layer heads (current default)
            self.head_sent = nn.Sequential(
                nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN // 2),
                nn.GELU(),
                nn.LayerNorm(HEAD_HIDDEN // 2),
                nn.Dropout(HEAD_DROPOUT * 0.8),
                nn.Linear(HEAD_HIDDEN // 2, HEAD_HIDDEN // 4),
                nn.GELU(),
                nn.LayerNorm(HEAD_HIDDEN // 4),
                nn.Dropout(HEAD_DROPOUT * 0.7),
                nn.Linear(HEAD_HIDDEN // 4, num_sent)
            )
            self.head_pol = nn.Sequential(
                nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN // 2),
                nn.GELU(),
                nn.LayerNorm(HEAD_HIDDEN // 2),
                nn.Dropout(HEAD_DROPOUT * 0.8),
                nn.Linear(HEAD_HIDDEN // 2, HEAD_HIDDEN // 4),
                nn.GELU(),
                nn.LayerNorm(HEAD_HIDDEN // 4),
                nn.Dropout(HEAD_DROPOUT * 0.7),
                nn.Linear(HEAD_HIDDEN // 4, num_pol)
            )
        elif HEAD_LAYERS == 2:
            # 2-layer heads (legacy)
            self.head_sent = nn.Sequential(
                nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN // 2),
                nn.GELU(),
                nn.LayerNorm(HEAD_HIDDEN // 2),
                nn.Dropout(HEAD_DROPOUT * 0.8),
                nn.Linear(HEAD_HIDDEN // 2, num_sent)
            )
            self.head_pol = nn.Sequential(
                nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN // 2),
                nn.GELU(),
                nn.LayerNorm(HEAD_HIDDEN // 2),
                nn.Dropout(HEAD_DROPOUT * 0.8),
                nn.Linear(HEAD_HIDDEN // 2, num_pol)
            )
        else:
            # Single-layer heads (fallback)
            self.head_sent = nn.Linear(HEAD_HIDDEN, num_sent)
            self.head_pol  = nn.Linear(HEAD_HIDDEN, num_pol)

        # Enable gradient checkpointing if configured
        if USE_GRADIENT_CHECKPOINTING:
            self.encoder.gradient_checkpointing_enable()

    def _pool(self, outputs, attention_mask):
        """
        Flexible representation pooling with multiple strategies.
        """
        # 🔥 NEW: Attention-based pooling (learns importance)
        if REP_POOLING == "attention":
            return self.attention_pool(outputs.hidden_states, attention_mask)

        # Pooler output (if available)
        if REP_POOLING == "pooler" and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output

        # CLS token
        if REP_POOLING == "cls":
            return outputs.last_hidden_state[:, 0]

        # Default: last4_mean (average of last 4 layers)
        hs = outputs.hidden_states  # tuple of [layer0..last]
        last4 = torch.stack(hs[-4:]).mean(dim=0)       # [B, T, H]
        return mean_pooling(last4, attention_mask)     # [B, H]

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                sentiment_labels=None,
                polarization_labels=None):
        # Encode input
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
            output_hidden_states=True  # Always need hidden states now
        )

        # Pool encoder outputs
        pooled = self._pool(outputs, attention_mask)

        # Project to HEAD_HIDDEN dimension
        z = self.trunk_proj(pooled)

        # Pass through residual blocks
        for block in self.trunk_blocks:
            z = block(z)

        # Final trunk normalization and dropout
        z = self.trunk_norm(z)
        z = self.trunk_dropout(z)

        # 🔥 NEW: Task-specific feature extraction
        z_sent = self.sent_proj(z)  # Sentiment-specific features
        z_pol = self.pol_proj(z)    # Polarization-specific features

        # Pass through task heads
        if HEAD_LAYERS >= 4 and isinstance(self.head_sent, nn.ModuleList):
            # ModuleList: apply each layer sequentially
            for layer in self.head_sent:
                z_sent = layer(z_sent)
            for layer in self.head_pol:
                z_pol = layer(z_pol)
            sent_logits = z_sent
            pol_logits = z_pol
        else:
            # Sequential: apply directly
            sent_logits = self.head_sent(z_sent)
            pol_logits = self.head_pol(z_pol)

        return {"logits": (sent_logits, pol_logits)}

```

# SECTION 7

```python
# SECTION 7

def compute_metrics_multi(eval_pred):
    (sent_logits, pol_logits) = eval_pred.predictions
    (y_sent, y_pol) = eval_pred.label_ids

    ps = np.argmax(sent_logits, axis=1)
    pp = np.argmax(pol_logits, axis=1)

    # Macro metrics
    sent_report = classification_report(y_sent, ps, output_dict=True, zero_division=0)
    pol_report  = classification_report(y_pol,  pp, output_dict=True, zero_division=0)

    sent_f1 = sent_report["macro avg"]["f1-score"]
    pol_f1  = pol_report["macro avg"]["f1-score"]
    macro_f1_avg = (sent_f1 + pol_f1) / 2.0

    return {
        "sent_acc": sent_report["accuracy"],
        "sent_prec": sent_report["macro avg"]["precision"],
        "sent_rec": sent_report["macro avg"]["recall"],
        "sent_f1": sent_f1,

        "pol_acc": pol_report["accuracy"],
        "pol_prec": pol_report["macro avg"]["precision"],
        "pol_rec": pol_report["macro avg"]["recall"],
        "pol_f1": pol_f1,

        "macro_f1_avg": macro_f1_avg
    }

```

# SECTION 8

```python
# ===== Section 8 — Custom Trainer (R-Drop + LLRD + safe prediction_step) =====
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from torch.utils.data import DataLoader
from torch.optim import AdamW

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        loss = F.nll_loss((1 - p) ** self.gamma * logp, target, weight=self.weight, reduction="none")
        return loss.mean() if self.reduction == "mean" else loss.sum()

def _sym_kl_with_logits(logits1, logits2):
    p = F.log_softmax(logits1, dim=-1);  q = F.log_softmax(logits2, dim=-1)
    p_exp, q_exp = p.exp(), q.exp()
    return 0.5 * (F.kl_div(p, q_exp, reduction="batchmean") + F.kl_div(q, p_exp, reduction="batchmean"))

class MultiTaskTrainer(Trainer):
    def __init__(self, *args, class_weights=None, task_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights or {}
        self.task_weights  = task_weights or {"sentiment": 1.0, "polarization": 1.0}
        self._custom_train_sampler = None

    # ----- LLRD optimizer -----
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        if not USE_LLRD:
            self.optimizer = AdamW(self.get_decay_parameter_groups(self.model), lr=LR, weight_decay=WEIGHT_DECAY)
            return self.optimizer

        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        encoder = self.model.encoder
        n_layers = getattr(encoder.config, "num_hidden_layers", 12)
        # Try to access sequential layers
        layers = getattr(getattr(encoder, "encoder", encoder), "layer", None)
        if layers is None:
            # Fallback: no LLRD if we can't find layers
            self.optimizer = AdamW(self.get_decay_parameter_groups(self.model), lr=LR, weight_decay=WEIGHT_DECAY)
            return self.optimizer

        param_groups = []

        # Embeddings (lowest lr)
        emb = getattr(encoder, "embeddings", None)
        if emb is not None:
            lr_emb = LR * (LLRD_DECAY ** n_layers)
            decay, nodecay = [], []
            for n, p in emb.named_parameters():
                (nodecay if any(nd in n for nd in no_decay) else decay).append(p)
            if decay:   param_groups.append({"params": decay,   "lr": lr_emb, "weight_decay": WEIGHT_DECAY})
            if nodecay: param_groups.append({"params": nodecay, "lr": lr_emb, "weight_decay": 0.0})

        # Encoder blocks (increasing LR toward the top)
        for i in range(n_layers):
            block = layers[i]
            lr_i = LR * (LLRD_DECAY ** (n_layers - 1 - i))
            decay, nodecay = [], []
            for n, p in block.named_parameters():
                (nodecay if any(nd in n for nd in no_decay) else decay).append(p)
            if decay:   param_groups.append({"params": decay,   "lr": lr_i, "weight_decay": WEIGHT_DECAY})
            if nodecay: param_groups.append({"params": nodecay, "lr": lr_i, "weight_decay": 0.0})

        # Pooler (if any)
        pooler = getattr(encoder, "pooler", None)
        if pooler is not None:
            decay, nodecay = [], []
            for n, p in pooler.named_parameters():
                (nodecay if any(nd in n for nd in no_decay) else decay).append(p)
            if decay:   param_groups.append({"params": decay,   "lr": LR, "weight_decay": WEIGHT_DECAY})
            if nodecay: param_groups.append({"params": nodecay, "lr": LR, "weight_decay": 0.0})

        # 🔥 RUN #14: Heads/trunk (highest LR) - UPDATED FOR NEW ARCHITECTURE
        head_lr = LR * HEAD_LR_MULT

        # Collect all head modules (updated for new architecture)
        head_modules = []

        # Add trunk components (new architecture with residual blocks)
        if hasattr(self.model, 'trunk_proj'):
            head_modules.append(self.model.trunk_proj)
        if hasattr(self.model, 'trunk_blocks'):
            head_modules.extend(list(self.model.trunk_blocks))
        if hasattr(self.model, 'trunk_norm'):
            head_modules.append(self.model.trunk_norm)
        if hasattr(self.model, 'trunk_dropout'):
            head_modules.append(self.model.trunk_dropout)

        # Add task-specific projections (new in Run #14)
        if hasattr(self.model, 'sent_proj'):
            head_modules.append(self.model.sent_proj)
        if hasattr(self.model, 'pol_proj'):
            head_modules.append(self.model.pol_proj)

        # Add attention pooling (new in Run #14)
        if hasattr(self.model, 'attention_pool'):
            head_modules.append(self.model.attention_pool)

        # Add task heads
        head_modules.append(self.model.head_sent)
        head_modules.append(self.model.head_pol)

        # Collect parameters with decay/no-decay split
        decay, nodecay = [], []
        for m in head_modules:
            for n, p in m.named_parameters():
                (nodecay if any(nd in n for nd in no_decay) else decay).append(p)

        if decay:   param_groups.append({"params": decay,   "lr": head_lr, "weight_decay": WEIGHT_DECAY})
        if nodecay: param_groups.append({"params": nodecay, "lr": head_lr, "weight_decay": 0.0})

        self.optimizer = AdamW(param_groups, lr=LR)  # lr here is ignored per-group
        return self.optimizer

    def set_train_sampler(self, sampler):
        self._custom_train_sampler = sampler

    def get_train_dataloader(self):
        if self.train_dataset is None:
            return None
        if self._custom_train_sampler is not None:
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=self._custom_train_sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        return super().get_train_dataloader()

    def _sent_loss_fn(self, weight, logits, target):
        if USE_FOCAL_SENTIMENT:
            return FocalLoss(weight=weight, gamma=FOCAL_GAMMA_SENTIMENT)(logits, target)
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=float(LABEL_SMOOTH_SENTIMENT))(logits, target)

    def _pol_loss_fn(self, weight, logits, target):
        if USE_FOCAL_POLARITY:
            return FocalLoss(weight=weight, gamma=FOCAL_GAMMA_POLARITY)(logits, target)
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=float(LABEL_SMOOTH_POLARITY))(logits, target)

    def compute_loss(self, model, inputs, return_outputs=False):
        y_sent = inputs.pop("sentiment_labels")
        y_pol  = inputs.pop("polarization_labels")

        # R-Drop with warmup: two forward passes with dropout
        current_epoch = getattr(self.state, 'epoch', 0) if hasattr(self, 'state') else 0
        use_rdrop_now = USE_RDROP and model.training and current_epoch >= RDROP_WARMUP_EPOCHS

        if use_rdrop_now:
            outputs1 = model(**inputs)
            outputs2 = model(**inputs)
            s1, p1 = outputs1["logits"]
            s2, p2 = outputs2["logits"]

            ws = self.class_weights.get("sentiment", None); ws = ws.to(s1.device) if ws is not None else None
            wp = self.class_weights.get("polarization", None); wp = wp.to(p1.device) if wp is not None else None

            ce_s = 0.5 * (self._sent_loss_fn(ws, s1, y_sent) + self._sent_loss_fn(ws, s2, y_sent))
            ce_p = 0.5 * (self._pol_loss_fn(wp,  p1, y_pol)  + self._pol_loss_fn(wp,  p2, y_pol))
            kl_s = _sym_kl_with_logits(s1, s2)
            kl_p = _sym_kl_with_logits(p1, p2)

            w_s = float(self.task_weights.get("sentiment", 1.0))
            w_p = float(self.task_weights.get("polarization", 1.0))

            # Gradual R-Drop alpha rampup for stability
            rdrop_factor = min(1.0, (current_epoch - RDROP_WARMUP_EPOCHS + 1) / 2.0)
            loss = w_s * ce_s + w_p * ce_p + (RDROP_ALPHA * rdrop_factor) * (kl_s + kl_p)
            if return_outputs:
                return loss, {"logits": (s1, p1)}
            return loss

        # Standard single forward
        outputs = model(**inputs)
        s, p = outputs["logits"]

        ws = self.class_weights.get("sentiment", None); ws = ws.to(s.device) if ws is not None else None
        wp = self.class_weights.get("polarization", None); wp = wp.to(p.device) if wp is not None else None

        loss_s = self._sent_loss_fn(ws, s, y_sent)
        loss_p = self._pol_loss_fn(wp, p, y_pol)

        w_s = float(self.task_weights.get("sentiment", 1.0))
        w_p = float(self.task_weights.get("polarization", 1.0))
        loss = w_s * loss_s + w_p * loss_p

        if return_outputs:
            outputs = dict(outputs); outputs["labels"] = (y_sent, y_pol)
            return loss, outputs
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Safe for inference (no labels provided)
        y_sent = inputs.get("sentiment_labels", None)
        y_pol  = inputs.get("polarization_labels", None)

        model_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        if "token_type_ids" in inputs:
            model_inputs["token_type_ids"] = inputs["token_type_ids"]

        model.eval()
        with torch.no_grad():
            outputs = model(**model_inputs)
            s, p = outputs["logits"]

        loss = None
        logits = (s.detach(), p.detach())
        labels = (y_sent, y_pol) if isinstance(y_sent, torch.Tensor) and isinstance(y_pol, torch.Tensor) else None
        return (loss, logits, labels)

# 🔴 RUN #8: Custom training_step REMOVED (R7 proved it's harmful!)
# Task-specific gradient control caused -7.91% regression (worst run ever)
# Restoring default Trainer behavior for gradient clipping

```


# SECTION 9

```python 
# ===== Section 9 — Train/Evaluate One Model (with grad accumulation) =====
from transformers import AutoTokenizer, DataCollatorWithPadding
import math, json, numpy as np, pandas as pd, os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import torch
from torch.utils.data import WeightedRandomSampler
from collections import Counter

def train_eval_one_model(model_key: str,
                         X_tr: pd.DataFrame, X_v: pd.DataFrame, X_te: pd.DataFrame,
                         ysent_tr: np.ndarray, ysent_v: np.ndarray, ysent_te: np.ndarray,
                         ypol_tr: np.ndarray,  ypol_v: np.ndarray,  ypol_te: np.ndarray,
                         sent_w_np: np.ndarray, pol_w_np: np.ndarray):
    base_name = MODEL_CONFIGS[model_key]["name"]
    run_dir = os.path.join(OUT_DIR, f"{model_key}")
    os.makedirs(run_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_name)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    tr_titles, tr_texts = X_tr[TITLE_COL].values, X_tr[TEXT_COL].values
    v_titles,  v_texts  = X_v[TITLE_COL].values, X_v[TEXT_COL].values
    te_titles, te_texts = X_te[TITLE_COL].values, X_te[TEXT_COL].values

    train_ds = TaglishDataset(tr_titles, tr_texts, ysent_tr, ypol_tr, tokenizer, max_length=MAX_LENGTH)
    val_ds   = TaglishDataset(v_titles,  v_texts,  ysent_v,  ypol_v,  tokenizer, max_length=MAX_LENGTH)
    test_ds  = TaglishDataset(te_titles, te_texts, ysent_te, ypol_te, tokenizer, max_length=MAX_LENGTH)

    model = MultiTaskModel(base_name, num_sent_classes, num_pol_classes).to(device)

    sent_w = torch.tensor(sent_w_np, dtype=torch.float32)
    pol_w  = torch.tensor(pol_w_np,  dtype=torch.float32)

    args = make_training_args_compat(
        output_dir=run_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1_avg",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_dir=os.path.join(run_dir, "logs"),
        logging_steps=25,                    # More frequent logging
        logging_first_step=True,             # Log first step for debugging
        save_steps=500,                      # Save checkpoints more often
        eval_steps=None,                     # Eval at end of each epoch
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        dataloader_pin_memory=True,          # Performance optimization
        max_grad_norm=MAX_GRAD_NORM,         # ✅ R4 EXACT (global gradient clipping restored)
        label_smoothing_factor=0.0,          # We handle this in loss functions
        save_total_limit=5,                  # ✅ R4 EXACT
        prediction_loss_only=False           # Log all metrics
    )

    callbacks = get_early_stopping_callbacks(EARLY_STOP_PATIENCE)

    trainer = MultiTaskTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_multi,
        callbacks=callbacks,
        class_weights={"sentiment": sent_w, "polarization": pol_w},
        task_weights=TASK_LOSS_WEIGHTS
    )

    # ----- ENHANCED JOINT oversampling with objective boost -----
    if USE_OVERSAMPLING and USE_JOINT_OVERSAMPLING:
        pair_counts = Counter(zip(ysent_tr.tolist(), ypol_tr.tolist()))
        counts = np.array(list(pair_counts.values()), dtype=np.float32)
        med = float(np.median(counts)) if len(counts) else 1.0

        # Find objective class index (polarization)
        obj_idx = np.where(pol_le.classes_ == "objective")[0][0] if "objective" in pol_le.classes_ else 1

        # 🔥 NEW: Find neutral class index (sentiment)
        neutral_idx = np.where(sent_le.classes_ == "neutral")[0][0] if "neutral" in sent_le.classes_ else 1

        def inv_mult(c):
            if c <= 0: return JOINT_OVERSAMPLING_MAX_MULT
            return float(np.clip(med / float(c), 1.0, JOINT_OVERSAMPLING_MAX_MULT))

        inv_by_pair = {k: inv_mult(v) for k, v in pair_counts.items()}
        sample_weights = []

        for ys, yp in zip(ysent_tr, ypol_tr):
            inv = inv_by_pair.get((int(ys), int(yp)), 1.0)
            w = (1.0 - JOINT_ALPHA) * 1.0 + JOINT_ALPHA * inv

            # Smart oversampling: extra boost for objective class (polarization)
            if USE_SMART_OVERSAMPLING and int(yp) == obj_idx:
                w *= OBJECTIVE_BOOST_MULT

            # 🔥 NEW: Also boost neutral class (sentiment) - Fixes 49% F1!
            if USE_SMART_OVERSAMPLING and int(ys) == neutral_idx:
                w *= NEUTRAL_BOOST_MULT

            sample_weights.append(w)

        obj_boost_count = sum(1 for i, yp in enumerate(ypol_tr) if int(yp) == obj_idx and sample_weights[i] > 2.0)
        neutral_boost_count = sum(1 for i, ys in enumerate(ysent_tr) if int(ys) == neutral_idx and sample_weights[i] > 2.0)
        print(f"🔥 Enhanced Oversampling: min={min(sample_weights):.2f}, max={max(sample_weights):.2f}")
        print(f"   ├─ Objective boosted samples: {obj_boost_count} (target: weak class at 40% F1)")
        print(f"   └─ Neutral boosted samples: {neutral_boost_count} (target: weak class at 49% F1)")
        trainer.set_train_sampler(WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True))

    trainer.train()

    # Save validation logits and labels for ensembling
    try:
        val_out = trainer.predict(val_ds)
        np.save(os.path.join(run_dir, "val_sent_logits.npy"), val_out.predictions[0])
        np.save(os.path.join(run_dir, "val_pol_logits.npy"),  val_out.predictions[1])
        np.save(os.path.join(run_dir, "val_sent_labels.npy"),  ysent_v)
        np.save(os.path.join(run_dir, "val_pol_labels.npy"),   ypol_v)
    except Exception as e:
        print("[Warn] Could not save validation logits:", e)

    # Test
    test_out = trainer.predict(test_ds)
    metrics = {f"test_{k}": float(v) for k, v in test_out.metrics.items()}
    trainer.save_model(run_dir)
    tokenizer.save_pretrained(run_dir)
    with open(os.path.join(run_dir, "metrics_test.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    sent_logits, pol_logits = test_out.predictions
    ysent_pred = np.argmax(sent_logits, axis=1)
    ypol_pred  = np.argmax(pol_logits,  axis=1)

    cm_sent = confusion_matrix(ysent_te, ysent_pred, labels=list(range(num_sent_classes)))
    cm_pol  = confusion_matrix(ypol_te,  ypol_pred,  labels=list(range(num_pol_classes)))
    np.save(os.path.join(run_dir, "cm_sent.npy"), cm_sent)
    np.save(os.path.join(run_dir, "cm_pol.npy"),  cm_pol)

    def plot_cm(cm, labels, title, path_png):
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046); plt.tight_layout(); plt.savefig(path_png, dpi=160); plt.close(fig)

    plot_cm(cm_sent, sent_le.classes_, "Sentiment Confusion", os.path.join(run_dir, "cm_sent.png"))
    plot_cm(cm_pol,  pol_le.classes_,  "Polarization Confusion", os.path.join(run_dir, "cm_pol.png"))

    rep_sent = classification_report(ysent_te, ysent_pred, target_names=sent_le.classes_, digits=4, zero_division=0)
    rep_pol  = classification_report(ypol_te,  ypol_pred,  target_names=pol_le.classes_,  digits=4, zero_division=0)
    with open(os.path.join(run_dir, "report_sentiment.txt"), "w") as f: f.write(rep_sent)
    with open(os.path.join(run_dir, "report_polarization.txt"), "w") as f: f.write(rep_pol)

    return {"model_key": model_key, "base_name": base_name, **metrics}, (ysent_pred, ypol_pred)

# End timing for architecture setup
timer.end_section("SECTION 5-9: Model Architecture & Training Setup")

```

# SECTION 10

```python
# SECTION 10

timer.start_section("SECTION 10: Model Training Execution")

results = []
pred_cache = {}

for key in MODELS_TO_RUN:
    print(f"\n=== Running {key} -> {MODEL_CONFIGS[key]['name']} ===")
    row, preds = train_eval_one_model(
        key,
        X_train, X_val, X_test,
        ysent_train, ysent_val, ysent_test,
        ypol_train,  ypol_val,  ypol_test,
        sent_weights_np, pol_weights_np
    )
    results.append(row)
    pred_cache[key] = preds

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUT_DIR, "summary_results.csv"), index=False)

# End timing for training execution
timer.end_section("SECTION 10: Model Training Execution")
timer.start_section("SECTION 11+: Evaluation & Calibration")

results_df

```


## SECTION 10A

```python
# ============================================================================
# SECTION 10A — VERIFY ARTIFACTS & RESOLVE TOKENIZER + WEIGHTS (v2)
# Builds maps for: tokenizer_dir (usually run root) and weights_dir (checkpoint or run root).
# Run AFTER Section 10 (training) and BEFORE 11B/11C.
# ============================================================================

import os, re, json
from typing import Optional, Dict

def _has_weights(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "pytorch_model.bin")) or os.path.isfile(os.path.join(path, "model.safetensors"))

def _has_tokenizer(path: str) -> bool:
    # Minimal tokenizer files
    return (
        os.path.isfile(os.path.join(path, "tokenizer.json")) or
        os.path.isfile(os.path.join(path, "vocab.txt")) or
        os.path.isfile(os.path.join(path, "spiece.model"))
    )

def _list_checkpoints(run_dir: str):
    if not os.path.isdir(run_dir): return []
    chks = []
    for name in os.listdir(run_dir):
        p = os.path.join(run_dir, name)
        if os.path.isdir(p) and re.match(r"^checkpoint-\d+$", name):
            chks.append(p)
    # sort by gl

```

# SECTION 11

```python
# ===== Section 11 — Detailed Breakdown Reports (per-class + cross-slices) =====
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import os
import json

def per_class_breakdown(y_true, y_pred, class_names):
    rep = classification_report(
        y_true, y_pred,
        target_names=list(class_names),
        output_dict=True, zero_division=0
    )
    # Keep only the class rows in the given order
    rows = []
    for cname in class_names:
        if cname in rep:
            rows.append({
                "class": cname,
                "precision": rep[cname]["precision"],
                "recall":    rep[cname]["recall"],
                "f1":        rep[cname]["f1-score"],
                "support":   int(rep[cname]["support"]),
            })
        else:
            rows.append({"class": cname, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})
    return pd.DataFrame(rows)

def cross_slice_breakdown(
    slice_true,  # array of ints for the slicing label (e.g., true sentiment indices)
    slice_names, # names of the slicing label classes (e.g., sentiment class names)
    task_true,   # array of ints for the task we evaluate (e.g., true polarity indices)
    task_pred,   # array of ints for the task predictions (e.g., predicted polarity indices)
    task_names,  # names of the task classes (e.g., polarity class names)
    slice_label  # string for the slice axis name, e.g., "sentiment" or "polarity"
):
    """
    For each class s in slice_true, evaluate the task predictions on the subset where slice_true == s.
    Returns one row per slice value, including macro-F1, accuracy, and per-class F1 for the task.
    """
    rows = []
    for idx, sname in enumerate(slice_names):
        mask = (slice_true == idx)
        n = int(mask.sum())
        if n == 0:
            # No samples for this slice in test set
            row = {"slice": sname, "support": 0, "accuracy": np.nan, "macro_f1": np.nan}
            for tname in task_names:
                row[f"f1_{tname}"] = np.nan
            rows.append(row)
            continue

        rep = classification_report(
            task_true[mask], task_pred[mask],
            target_names=list(task_names),
            output_dict=True, zero_division=0
        )
        row = {
            "slice": sname,
            "support": n,
            "accuracy": rep["accuracy"],
            "macro_f1": rep["macro avg"]["f1-score"],
        }
        for tname in task_names:
            row[f"f1_{tname}"] = rep[tname]["f1-score"]
        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort slices by support (desc) for readability
    df = df.sort_values(by="support", ascending=False).reset_index(drop=True)
    return df

# Where to save things
DETAILS_DIR = os.path.join(OUT_DIR, "details")
os.makedirs(DETAILS_DIR, exist_ok=True)

all_breakdowns = {}

for key in MODELS_TO_RUN:
    print(f"\n=== Detailed breakdowns for {key} ===")
    ysent_pred, ypol_pred = pred_cache[key]

    # ---- Per-class reports on the full test set
    sent_per_class = per_class_breakdown(ysent_test, ysent_pred, sent_le.classes_)
    pol_per_class  = per_class_breakdown(ypol_test,  ypol_pred,  pol_le.classes_)

    # Save + show
    sent_csv = os.path.join(DETAILS_DIR, f"{key}_sentiment_per_class.csv")
    pol_csv  = os.path.join(DETAILS_DIR, f"{key}_polarization_per_class.csv")
    sent_per_class.to_csv(sent_csv, index=False)
    pol_per_class.to_csv(pol_csv, index=False)

    print("\nSentiment — per class (precision/recall/F1/support):")
    display(sent_per_class)

    print("\nPolarization — per class (precision/recall/F1/support):")
    display(pol_per_class)

    # ---- Cross-slice reports
    # Polarity performance within each (true) sentiment slice
    pol_given_sent = cross_slice_breakdown(
        slice_true=ysent_test, slice_names=sent_le.classes_,
        task_true=ypol_test,   task_pred=ypol_pred, task_names=pol_le.classes_,
        slice_label="sentiment"
    )
    pol_given_sent_csv = os.path.join(DETAILS_DIR, f"{key}_polarity_given_sentiment.csv")
    pol_given_sent.to_csv(pol_given_sent_csv, index=False)

    print("\nPolarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):")
    display(pol_given_sent)

    # Sentiment performance within each (true) polarity slice
    sent_given_pol = cross_slice_breakdown(
        slice_true=ypol_test,  slice_names=pol_le.classes_,
        task_true=ysent_test,  task_pred=ysent_pred, task_names=sent_le.classes_,
        slice_label="polarity"
    )
    sent_given_pol_csv = os.path.join(DETAILS_DIR, f"{key}_sentiment_given_polarity.csv")
    sent_given_pol.to_csv(sent_given_pol_csv, index=False)

    print("\nSentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):")
    display(sent_given_pol)

    # Keep for a single JSON bundle if you like
    all_breakdowns[key] = {
        "sentiment_per_class_csv": sent_csv,
        "polarization_per_class_csv": pol_csv,
        "polarity_given_sentiment_csv": pol_given_sent_csv,
        "sentiment_given_polarity_csv": sent_given_pol_csv
    }

# Optional: write an index JSON pointing to all CSVs
with open(os.path.join(DETAILS_DIR, "index.json"), "w") as f:
    json.dump(all_breakdowns, f, indent=2)
print("\nSaved detailed breakdowns to:", DETAILS_DIR)
```

## SECTION 11A

```python
# ============================================================================
# SECTION 11C — MULTICLASS POLARITY CALIBRATION (v2)
# ============================================================================

from sklearn.metrics import classification_report
import numpy as np, json, os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, TrainingArguments, DataCollatorWithPadding

# ============================================================================
# Helper Functions for Calibration
# ============================================================================

class _PlainPairDS(Dataset):
    """Simple dataset for inference-only (no labels needed)"""
    def __init__(self, titles, texts, tokenizer, max_length=224):
        self.titles, self.texts = list(titles), list(texts)
        self.tok = tokenizer
        self.max_length = max_length
        self.use_tt = "token_type_ids" in tokenizer.model_input_names

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tok(
            text=str(self.titles[idx]),
            text_pair=str(self.texts[idx]),
            truncation="only_second",
            max_length=self.max_length,
            return_token_type_ids=self.use_tt
        )

def _get_pol_logits(model_key, titles, texts):
    """Get polarization logits from trained model"""
    # Load tokenizer and model
    run_dir = os.path.join(OUT_DIR, model_key)
    model_name = MODEL_CONFIGS[model_key]["name"]

    print(f"   Loading model from: {run_dir}")
    tokenizer = AutoTokenizer.from_pretrained(run_dir if os.path.exists(os.path.join(run_dir, "tokenizer.json")) else model_name)

    # Rebuild model and load weights
    model = MultiTaskModel(model_name, num_sent_classes, num_pol_classes)

    # 🔧 RUN #9 FIX: Robust model loading - check multiple possible weight file formats
    # Modern Transformers saves in different formats (pytorch_model.bin, model.safetensors, checkpoint-*/pytorch_model.bin)
    weights_loaded = False

    # Try 1: pytorch_model.bin in root
    model_file = os.path.join(run_dir, "pytorch_model.bin")
    if os.path.exists(model_file):
        print(f"   ✓ Loading weights from: {model_file}")
        model.load_state_dict(torch.load(model_file, map_location=device), strict=False)
        weights_loaded = True

    # Try 2: model.safetensors in root
    if not weights_loaded:
        model_file = os.path.join(run_dir, "model.safetensors")
        if os.path.exists(model_file):
            print(f"   ✓ Loading weights from: {model_file}")
            from safetensors.torch import load_file
            state_dict = load_file(model_file)
            model.load_state_dict(state_dict, strict=False)
            weights_loaded = True

    # Try 3: Latest checkpoint subdirectory (checkpoint-XXXX/pytorch_model.bin)
    if not weights_loaded:
        checkpoints = [d for d in os.listdir(run_dir) if d.startswith('checkpoint-') and os.path.isdir(os.path.join(run_dir, d))]
        if checkpoints:
            # Get the latest checkpoint by number
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
            checkpoint_dir = os.path.join(run_dir, latest_checkpoint)
            model_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
            if os.path.exists(model_file):
                print(f"   ✓ Loading weights from checkpoint: {checkpoint_dir}")
                model.load_state_dict(torch.load(model_file, map_location=device), strict=False)
                weights_loaded = True

    if not weights_loaded:
        print(f"   ⚠️ WARNING: No trained weights found in {run_dir}")
        print(f"      Checked: pytorch_model.bin, model.safetensors, checkpoint-*/pytorch_model.bin")
        print(f"      Using UNTRAINED model - calibration will be ineffective!")

    model.to(device)
    model.eval()

    # Create dataset and trainer
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    args = TrainingArguments(
        output_dir=os.path.join(run_dir, "calib_tmp"),
        per_device_eval_batch_size=64,
        report_to="none"
    )

    dummy_trainer = MultiTaskTrainer(
        model=model,
        args=args,
        data_collator=collator,
        class_weights=None,
        task_weights=None
    )

    ds = _PlainPairDS(titles, texts, tokenizer, MAX_LENGTH)
    out = dummy_trainer.predict(ds)
    _, pol_logits = out.predictions

    return pol_logits

# ============================================================================
# Calibration Functions
# ============================================================================

def coord_search_biases(pol_logits_val, y_val, class_names, passes=2, grid=(-0.8, 0.8, 0.1)):
    lo, hi, step = grid
    C = pol_logits_val.shape[1]
    b = np.zeros(C, dtype=np.float32)

    def macro_f1_with(bias_vec):
        y_pred = np.argmax(pol_logits_val + bias_vec.reshape(1, -1), axis=1)
        rep = classification_report(y_val, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        return rep["macro avg"]["f1-score"]

    best = macro_f1_with(b)
    for _ in range(passes):
        improved = False
        for c in range(C):
            best_b_c, best_score_c = b[c], best
            for val in np.arange(lo, hi + 1e-9, step):
                b_try = b.copy()
                b_try[c] = val
                score = macro_f1_with(b_try)
                if score > best_score_c + 1e-6:
                    best_score_c, best_b_c = score, val
            if best_b_c != b[c]:
                b[c] = best_b_c
                best = best_score_c
                improved = True
        if not improved:
            break
    return b, float(best)

CALIB_DIR2 = os.path.join(OUT_DIR, "calibration_vector")
os.makedirs(CALIB_DIR2, exist_ok=True)

print("🎯 MULTICLASS CALIBRATION - Optimize prediction biases for better performance")
print("="*70)

for key in MODELS_TO_RUN:
    print(f"\n🔧 Calibrating {key} ({MODEL_CONFIGS[key]['name']})...")

    print(f"📊 Step 1: Extracting polarization logits from trained model...")
    pol_val_logits = _get_pol_logits(key, X_val[TITLE_COL].values,  X_val[TEXT_COL].values)
    pol_tst_logits = _get_pol_logits(key, X_test[TITLE_COL].values, X_test[TEXT_COL].values)
    print(f"   ✓ Validation logits shape: {pol_val_logits.shape}")
    print(f"   ✓ Test logits shape: {pol_tst_logits.shape}")

    y_val = ypol_val
    y_tst = ypol_test
    class_names = list(pol_le.classes_)

    print(f"🔍 Step 2: Searching for optimal bias vector (coordinate search)...")
    b_vec, val_macro = coord_search_biases(pol_val_logits, y_val, class_names, passes=3, grid=(-0.8, 0.8, 0.1))
    print(f"   ✓ Optimal bias vector found (VAL macro-F1={val_macro:.3f}):")
    for cname, bias_val in zip(class_names, b_vec):
        print(f"      • {cname:>13}: {bias_val:+.2f}")

    # Test before/after
    print(f"📈 Step 3: Evaluating calibration impact on test set...")
    y_before = np.argmax(pol_tst_logits, axis=1)
    rep_before = classification_report(y_tst, y_before, target_names=class_names, output_dict=True, zero_division=0)

    y_after = np.argmax(pol_tst_logits + b_vec.reshape(1, -1), axis=1)
    rep_after  = classification_report(y_tst, y_after, target_names=class_names, output_dict=True, zero_division=0)

    improvement = rep_after['macro avg']['f1-score'] - rep_before['macro avg']['f1-score']
    print(f"\n   📊 TEST MACRO-F1: {rep_before['macro avg']['f1-score']:.3f} → {rep_after['macro avg']['f1-score']:.3f} ({improvement:+.3f})\n")
    print("   Per-class breakdown:")
    for cname in class_names:
        b = rep_before[cname]; a = rep_after[cname]
        f1_change = a['f1-score'] - b['f1-score']
        emoji = "📈" if f1_change > 0 else "📉" if f1_change < 0 else "➡️"
        print(f"   {emoji} {cname:>13}: P={b['precision']:.3f} R={b['recall']:.3f} F1={b['f1-score']:.3f} (n={int(b['support'])})"
              f"  →  P={a['precision']:.3f} R={a['recall']:.3f} F1={a['f1-score']:.3f} ({f1_change:+.3f})")

    # Save calibration results
    calib_file = os.path.join(CALIB_DIR2, f"{key}_bias_vector.json")
    with open(calib_file, "w") as f:
        json.dump({
            "bias_vector": {class_names[i]: float(b_vec[i]) for i in range(len(class_names))},
            "val_macro_f1": val_macro,
            "test_macro_f1_before": float(rep_before["macro avg"]["f1-score"]),
            "test_macro_f1_after":  float(rep_after["macro avg"]["f1-score"])
        }, f, indent=2)

    print(f"\n✅ Calibration complete! Bias vector saved to:")
    print(f"   {calib_file}")

print(f"\n{'='*70}")
print(f"🎉 CALIBRATION FINISHED - All models optimized!")
```

# SECTION 12

```python
# ===== Section 12 — Length Diagnostics (clean) =====
import warnings

def token_lengths_summary(texts, titles, tokenizer, n=5000):
    # Random sample (or full if dataset is small)
    n = min(n, len(texts))
    idx = np.random.choice(len(texts), size=n, replace=False) if len(texts) > n else np.arange(len(texts))

    lengths = []
    # Silence the "sequence > 512" warnings emitted by some tokenizers for inspection
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Token indices sequence length is longer.*")
        for i in idx:
            s = f"{titles[i]} [SEP] {texts[i]}"
            # We want raw length pre-truncation to choose MAX_LENGTH wisely
            ids = tokenizer.encode(s, add_special_tokens=True, truncation=False)
            lengths.append(len(ids))

    arr = np.array(lengths)
    stats = {
        "mean": float(arr.mean()),
        "p50":  float(np.percentile(arr, 50)),
        "p90":  float(np.percentile(arr, 90)),
        "p95":  float(np.percentile(arr, 95)),
        "p99":  float(np.percentile(arr, 99)),
        "max":  int(arr.max())
    }
    print("Token length stats:", stats)
    return stats

for key in MODELS_TO_RUN:
    name = MODEL_CONFIGS[key]["name"]
    tok = AutoTokenizer.from_pretrained(name)
    print(f"\n[{key}] {name}")
    token_lengths_summary(
        texts=X_train[TEXT_COL].values,
        titles=X_train[TITLE_COL].values,
        tokenizer=tok,
        n=5000
    )

# Tip:
# If p95 is comfortably < 192, you're fine. If you see p95 > 192, consider MAX_LENGTH=224
# (Update in Section 3 if you decide to bump it.)

# Final timing summary
timer.end_section("SECTION 11+: Evaluation & Calibration")
timer.get_summary()
```
