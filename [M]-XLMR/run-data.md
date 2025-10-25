🚀 Starting SECTION 10: Model Training Execution...

=== Running xlm_roberta -> xlm-roberta-base ===
🔥 Enhanced Oversampling: min=0.30, max=14.88
├─ Objective boosted samples: 405 (target: weak class at 40% F1)
└─ Neutral boosted samples: 227 (target: weak class at 49% F1)
[2616/2900 1:56:46 < 12:41, 0.37 it/s, Epoch 18/20]
Epoch Training Loss Validation Loss Sent Acc Sent Prec Sent Rec Sent F1 Pol Acc Pol Prec Pol Rec Pol F1 Macro F1 Avg
0 1.155800 No log 0.367893 0.282820 0.433994 0.276111 0.078261 0.135721 0.350900 0.076173 0.176142
1 0.834200 No log 0.577926 0.529158 0.601030 0.539341 0.351171 0.449856 0.505657 0.330179 0.434760
3 0.483900 No log 0.572575 0.619408 0.563941 0.444907 0.626087 0.535490 0.629994 0.546533 0.495720
4 0.323200 No log 0.648829 0.601048 0.662762 0.605832 0.571906 0.530321 0.638512 0.516856 0.561344
6 0.213700 No log 0.653512 0.615538 0.681717 0.632965 0.686957 0.572490 0.655287 0.595128 0.614047
7 0.182400 No log 0.686288 0.637945 0.687515 0.656296 0.658194 0.571895 0.623153 0.574528 0.615412
9 0.140800 No log 0.701672 0.679395 0.672425 0.675811 0.756522 0.607293 0.626800 0.615681 0.645746
10 0.114400 No log 0.726421 0.697234 0.666821 0.670653 0.754515 0.630271 0.598972 0.606730 0.638691
12 0.096500 No log 0.724415 0.691180 0.690762 0.689071 0.763211 0.628208 0.619023 0.621387 0.655229
13 0.087800 No log 0.731104 0.697698 0.683201 0.683660 0.752508 0.618023 0.619208 0.616809 0.650235
15 0.076700 No log 0.740468 0.723447 0.678595 0.690950 0.774582 0.639972 0.605446 0.617978 0.654464
16 0.068100 No log 0.731104 0.702551 0.687120 0.692609 0.720401 0.611973 0.607166 0.595484 0.644047
18 0.077700 No log 0.731773 0.702724 0.683998 0.690199 0.751839 0.627625 0.608471 0.608286 0.649242
✅ SECTION 10: Model Training Execution completed in 2.0h 58m
🕒 Total runtime so far: 4.4h 26m

---

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key base_name test_test_sent_acc test_test_sent_prec test_test_sent_rec test_test_sent_f1 test_test_pol_acc test_test_pol_prec test_test_pol_rec test_test_pol_f1 test_test_macro_f1_avg test_test_runtime test_test_samples_per_second test_test_steps_per_second
0 xlm_roberta xlm-roberta-base 0.759866 0.733486 0.716192 0.719279 0.757191 0.666949 0.661103 0.663085 0.691182 5.4672 273.45 17.194

=== Detailed breakdowns for xlm_roberta ===

Sentiment — per class (precision/recall/F1/support):
class precision recall f1 support
0 negative 0.792893 0.881490 0.834848 886
1 neutral 0.650685 0.473815 0.548341 401
2 positive 0.756881 0.793269 0.774648 208

Polarization — per class (precision/recall/F1/support):
class precision recall f1 support
0 non_polarized 0.615063 0.675862 0.644031 435
1 objective 0.536585 0.488889 0.511628 90
2 partisan 0.849198 0.818557 0.833596 970

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice support accuracy macro_f1 f1_non_polarized f1_objective f1_partisan
0 negative 886 0.811512 0.629113 0.594595 0.409091 0.883652
1 neutral 401 0.640898 0.618425 0.671679 0.554455 0.629139
2 positive 208 0.750000 0.673709 0.694444 0.518519 0.808163

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice support accuracy macro_f1 f1_negative f1_neutral f1_positive
0 partisan 970 0.805155 0.678941 0.880111 0.361345 0.795367
1 non_polarized 435 0.678161 0.696960 0.695652 0.619718 0.775510
2 objective 90 0.666667 0.613333 0.600000 0.740000 0.500000

Saved detailed breakdowns to: ./runs_xlm_roberta_optimized/details

# 🎯 MULTICLASS CALIBRATION - Optimize prediction biases for better performance

🔧 Calibrating xlm_roberta (xlm-roberta-base)...
📊 Step 1: Extracting polarization logits from trained model...
Loading model from: ./runs_xlm_roberta_optimized/xlm_roberta
Warning: No trained weights found at ./runs_xlm_roberta_optimized/xlm_roberta/pytorch_model.bin, using untrained model
Loading model from: ./runs_xlm_roberta_optimized/xlm_roberta
Warning: No trained weights found at ./runs_xlm_roberta_optimized/xlm_roberta/pytorch_model.bin, using untrained model
✓ Validation logits shape: (1495, 3)
✓ Test logits shape: (1495, 3)
🔍 Step 2: Searching for optimal bias vector (coordinate search)...
✓ Optimal bias vector found (VAL macro-F1=0.375):
• non_polarized: +0.60
• objective: -0.10
• partisan: +0.30
📈 Step 3: Evaluating calibration impact on test set...

📊 TEST MACRO-F1: 0.150 → 0.150 (+0.000)

Per-class breakdown:
➡️ non_polarized: P=0.291 R=1.000 F1=0.451 (n=435) → P=0.291 R=1.000 F1=0.451 (+0.000)
➡️ objective: P=0.000 R=0.000 F1=0.000 (n=90) → P=0.000 R=0.000 F1=0.000 (+0.000)
➡️ partisan: P=0.000 R=0.000 F1=0.000 (n=970) → P=0.000 R=0.000 F1=0.000 (+0.000)

✅ Calibration complete! Bias vector saved to:
./runs_xlm_roberta_optimized/calibration_vector/xlm_roberta_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!

[xlm_roberta] xlm-roberta-base
Token indices sequence length is longer than the specified maximum sequence length for this model (950 > 512). Running this sequence through the model will result in indexing errors
Token length stats: {'mean': 106.9514, 'p50': 96.0, 'p90': 170.0, 'p95': 182.0, 'p99': 215.0, 'max': 950}
✅ SECTION 11+: Evaluation & Calibration completed in 54.5s
🕒 Total runtime so far: 4.5h 27m

---

============================================================
⏱️ EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports : 16.0s
SECTION 3: Configuration Setup : 13.8m 45s
SECTION 4: Data Loading & Preprocessing : 0.1s
SECTION 5-9: Model Architecture & Training Setup : 9.1s
SECTION 10: Model Training Execution : 2.0h 58m
SECTION 11+: Evaluation & Calibration : 54.5s
======================================== : ==========
TOTAL EXECUTION TIME : 4.5h 27m
============================================================
