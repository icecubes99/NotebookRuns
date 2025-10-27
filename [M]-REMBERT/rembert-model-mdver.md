# SECTION 1

```python
# ============================================================================
# SECTION 1: ENVIRONMENT SETUP (COLAB-FRIENDLY, LIGHTWEIGHT)
# ============================================================================
import sys, subprocess, importlib, os, random

def pipi(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "--no-cache-dir", *pkgs])

# Pin core libs (align with your mBERT/XLM-R stacks)
pipi(
    "numpy==2.1.1",
    "pandas==2.2.3",
    "scikit-learn==1.5.2",
    "matplotlib==3.9.2",
    "transformers==4.44.2",
    "accelerate==0.34.2",
)

import numpy as np
import torch, transformers, pandas as pd
print("CUDA:", torch.cuda.is_available())
```

# SECTION 2

```python
# ============================================================================
# SECTION 2: CONFIG + SEED (REM BERT BASELINE)
# ============================================================================
from transformers import set_seed

# IO
OUT_DIR = "./runs_rembert_optimized"
os.makedirs(OUT_DIR, exist_ok=True)

# Data
CSV_PATH     = "/content/adjudications_2025-10-22.csv"              # base
AUG_CSV_PATH = "/content/augmented_adjudications_2025-10-22.csv"    # augmented
USE_AUGMENTED_TRAIN = True  # append augmentation to TRAIN only

# Columns (mirror mBERT notebook)
TITLE_COL = "Title"
TEXT_COL  = "Comment"
SENT_COL  = "Final Sentiment"
POL_COL   = "Final Polarization"

# Model
MODEL_NAME = "google/rembert"
MAX_LENGTH = 320
USE_GRADIENT_CHECKPOINTING = True

# Train
EPOCHS = 20
BATCH_SIZE = 12
GRAD_ACCUM_STEPS = 3
LR = 2.0e-5              # safe starting LR for RemBERT
WARMUP_RATIO = 0.20
WEIGHT_DECAY = 0.03
EARLY_STOP_PATIENCE = 8
MAX_GRAD_NORM = 0.5

# Heads / pooling (kept simple; mirrors your Run 16)
HEAD_HIDDEN = 896
HEAD_LAYERS = 3
HEAD_DROPOUT = 0.28
REP_POOLING = "last4_mean"
HEAD_LR_MULT = 3.0

# Loss / task weights
FOCAL_GAMMA_SENTIMENT = 2.5
FOCAL_GAMMA_POLARITY  = 3.5
LABEL_SMOOTH_SENTIMENT = 0.10
LABEL_SMOOTH_POLARITY  = 0.08
TASK_LOSS_WEIGHTS = {"sentiment": 1.0, "polarization": 1.4}

# LLRD
USE_LLRD = True
LLRD_DECAY = 0.90

# Seed
SEED = 42

def seed_all(seed=SEED):
    transformers.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_all(SEED)
print(f"Seed set: {SEED}")
```
os.environ["WANDB_DISABLED"] = "true"  # disable W&B autologging

# SECTION 3

```python
# ============================================================================
# SECTION 3: DATA LOADING + STRATIFIED SPLIT + AUGMENT APPEND (TRAIN ONLY)
# ============================================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load base CSV (split on base only)
df = pd.read_csv(CSV_PATH)
required = [TITLE_COL, TEXT_COL, SENT_COL, POL_COL]
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns: {missing}. Available: {list(df.columns)}")

df = df.dropna(subset=required).reset_index(drop=True)

# Encode labels
sent_le = LabelEncoder().fit(df[SENT_COL].astype(str))
pol_le  = LabelEncoder().fit(df[POL_COL].astype(str))

df["sent_y"], df["pol_y"] = sent_le.transform(df[SENT_COL].astype(str)), pol_le.transform(df[POL_COL].astype(str))

# Stratify on joint so both tasks are balanced
joint = df["sent_y"] * 10 + df["pol_y"]
X = df[[TITLE_COL, TEXT_COL]].copy()
y_sent = df["sent_y"].values
y_pol  = df["pol_y"].values

X_train, X_tmp, ysent_train, ysent_tmp, ypol_train, ypol_tmp = train_test_split(
    X, y_sent, y_pol, test_size=0.30, random_state=SEED, stratify=joint
)
joint_tmp = ysent_tmp * 10 + ypol_tmp
X_val, X_test, ysent_val, ysent_test, ypol_val, ypol_test = train_test_split(
    X_tmp, ysent_tmp, ypol_tmp, test_size=0.50, random_state=SEED, stratify=joint_tmp
)

# Append augmentation to TRAIN only
if USE_AUGMENTED_TRAIN and os.path.isfile(AUG_CSV_PATH):
    aug = pd.read_csv(AUG_CSV_PATH).dropna(subset=required)
    aug["sent_y"], aug["pol_y"] = sent_le.transform(aug[SENT_COL].astype(str)), pol_le.transform(aug[POL_COL].astype(str))
    X_train = pd.concat([X_train, aug[[TITLE_COL, TEXT_COL]]], ignore_index=True)
    ysent_train = np.concatenate([ysent_train, aug["sent_y"].values])
    ypol_train  = np.concatenate([ypol_train,  aug["pol_y"].values])
    print("Augmentation appended to TRAIN only. Oversampling disabled.")
else:
    print("Augmented file missing or disabled; training on base only.")

print("Split sizes:", len(X_train), len(X_val), len(X_test))
```

# SECTION 4

```python
# ============================================================================
# SECTION 4: TOKENIZER + DATASETS
# ============================================================================
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, titles, texts, y_sent, y_pol, tok, max_length=320):
        self.titles = titles.reset_index(drop=True)
        self.texts  = texts.reset_index(drop=True)
        self.y_sent = y_sent
        self.y_pol  = y_pol
        self.tok = tok
        self.maxlen = max_length
    def __len__(self): return len(self.titles)
    def __getitem__(self, i):
        enc = self.tok(
            str(self.titles.iloc[i]),
            str(self.texts.iloc[i]),
            truncation=True,
            max_length=self.maxlen,
            padding=False,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["sentiment_labels"]    = torch.tensor(self.y_sent[i], dtype=torch.long)
        item["polarization_labels"] = torch.tensor(self.y_pol[i],  dtype=torch.long)
        return item

train_ds = PairDataset(X_train[TITLE_COL], X_train[TEXT_COL], ysent_train, ypol_train, tokenizer, MAX_LENGTH)
val_ds   = PairDataset(X_val[TITLE_COL],   X_val[TEXT_COL],   ysent_val,   ypol_val,   tokenizer, MAX_LENGTH)
test_ds  = PairDataset(X_test[TITLE_COL],  X_test[TEXT_COL],  ysent_test,  ypol_test,  tokenizer, MAX_LENGTH)
```

# SECTION 5

```python
# ============================================================================
# SECTION 5: MODEL (RemBERT backbone + simple multi-task heads)
# ============================================================================
import torch.nn as nn
from transformers import AutoModel

class MultiTaskModel(nn.Module):
    def __init__(self, base_model: str, num_sent: int, num_pol: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        if USE_GRADIENT_CHECKPOINTING:
            self.encoder.gradient_checkpointing_enable()
        self.hidden = self.encoder.config.hidden_size  # RemBERT: 1152
        # trunk
        self.trunk = nn.Sequential(
            nn.Linear(self.hidden, HEAD_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(HEAD_HIDDEN),
            nn.Dropout(HEAD_DROPOUT),
        )
        # heads
        def head_block(out_dim: int):
            return nn.Sequential(
                nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN // 2), nn.GELU(), nn.LayerNorm(HEAD_HIDDEN // 2), nn.Dropout(HEAD_DROPOUT*0.8),
                nn.Linear(HEAD_HIDDEN // 2, HEAD_HIDDEN // 4), nn.GELU(), nn.LayerNorm(HEAD_HIDDEN // 4), nn.Dropout(HEAD_DROPOUT*0.7),
                nn.Linear(HEAD_HIDDEN // 4, out_dim),
            )
        self.head_sent = head_block(num_sent)
        self.head_pol  = head_block(num_pol)

    def _rep(self, outputs, attention_mask):
        # last4_mean pooling
        hs = outputs.hidden_states  # tuple
        last4 = torch.stack(hs[-4:]).mean(0)  # [B,T,H]
        mask = attention_mask.unsqueeze(-1)
        return (last4 * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                sentiment_labels=None, polarization_labels=None):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
            output_hidden_states=True,
        )
        x = self._rep(out, attention_mask)
        x = self.trunk(x)
        return {"logits_sent": self.head_sent(x), "logits_pol": self.head_pol(x)}
```

# SECTION 6

```python
# ============================================================================
# SECTION 6: LOSS + TRAINER
# ============================================================================
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorWithPadding

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.weight, self.gamma, self.reduction = weight, gamma, reduction
    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        loss = F.nll_loss(((1 - p) ** self.gamma) * logp, target, weight=self.weight, reduction="none")
        return loss.mean() if self.reduction == "mean" else loss.sum() if self.reduction == "sum" else loss

class MTTrainer(Trainer):
    def __init__(self, task_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.task_weights = task_weights or {"sentiment": 1.0, "polarization": 1.0}
    def compute_loss(self, model, inputs, return_outputs=False):
        labels_s = inputs.pop("sentiment_labels"); labels_p = inputs.pop("polarization_labels")
        outputs = model(**inputs)
        ls, lp = outputs["logits_sent"], outputs["logits_pol"]
        loss_s = FocalLoss(gamma=FOCAL_GAMMA_SENTIMENT)(ls, labels_s)
        loss_p = FocalLoss(gamma=FOCAL_GAMMA_POLARITY)(lp, labels_p)
        loss = self.task_weights["sentiment"]*loss_s + self.task_weights["polarization"]*loss_p
        return (loss, outputs) if return_outputs else loss

num_sent, num_pol = int(df[SENT_COL].nunique()), int(df[POL_COL].nunique())
model = MultiTaskModel(MODEL_NAME, num_sent=num_sent, num_pol=num_pol)

# LLRD param groups
if USE_LLRD:
    n_layers = model.encoder.config.num_hidden_layers
    base_lr, decay = LR, LLRD_DECAY
    groups = []
    for i in range(n_layers):
        lr_i = base_lr * (decay ** (n_layers-1-i))
        params_i = [p for n,p in model.named_parameters() if f"encoder.layer.{i}." in n]
        if params_i: groups.append({"params": params_i, "lr": lr_i})
    emb_params = [p for n,p in model.named_parameters() if "embeddings" in n]
    if emb_params: groups.append({"params": emb_params, "lr": base_lr * (decay ** n_layers)})
    head_params = [p for n,p in model.named_parameters() if any(k in n for k in ["trunk", "head_sent", "head_pol"])]
    if head_params: groups.append({"params": head_params, "lr": base_lr * HEAD_LR_MULT})
    optim_params = groups
else:
    optim_params = model.parameters()

args = TrainingArguments(
    output_dir=os.path.join(OUT_DIR, "rembert"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    max_grad_norm=MAX_GRAD_NORM,
    report_to="none",
)

trainer = MTTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorWithPadding(tokenizer),
    task_weights=TASK_LOSS_WEIGHTS,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)]
)

print("Training…")
trainer.train()
print("Training complete.")
```

# SECTION 7

```python
# ============================================================================
# SECTION 7: EVALUATION (TEST) — QUICK REPORT
# ============================================================================
from sklearn.metrics import classification_report

model.eval()

def predict(ds):
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE)
    logits_s, logits_p, lab_s, lab_p = [], [], [], []
    for batch in loader:
        for k in ["input_ids","attention_mask","token_type_ids"]:
            if k in batch: batch[k] = batch[k].to(model.encoder.device)
        with torch.no_grad():
            out = model(**{k:batch[k] for k in ["input_ids","attention_mask","token_type_ids"] if k in batch})
        logits_s.append(out["logits_sent"].cpu()); logits_p.append(out["logits_pol"].cpu())
        lab_s.append(batch["sentiment_labels"]); lab_p.append(batch["polarization_labels"])
    logits_s = torch.cat(logits_s); logits_p = torch.cat(logits_p)
    y_s = torch.cat(lab_s).numpy(); y_p = torch.cat(lab_p).numpy()
    pred_s = logits_s.argmax(-1).numpy(); pred_p = logits_p.argmax(-1).numpy()
    return (y_s, pred_s), (y_p, pred_p)

(test_s, pred_s), (test_p, pred_p) = predict(test_ds)
print("=== SENTIMENT (test) ===\n", classification_report(test_s, pred_s, target_names=list(sent_le.classes_)))
print("=== POLARIZATION (test) ===\n", classification_report(test_p, pred_p, target_names=list(pol_le.classes_)))
```

# NOTES

```text
- Mirrors mBERT markdown structure, trimmed for RemBERT.
- Differences: base model = google/rembert, head capacity 896, dropout 0.28, LR 2e-5, augmentation-centric (no oversampling).
- Keep MAX_LENGTH=320; if OOM, lower batch size or sequence length.
- Add your calibration block after evaluation if desired.
```
