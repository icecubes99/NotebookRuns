🚀 Starting SECTION 10: Model Training Execution...

=== Running xlm_roberta -> xlm-roberta-base ===
tokenizer_config.json: 0%| | 0.00/25.0 [00:00<?, ?B/s]config.json: 0%| | 0.00/615 [00:00<?, ?B/s]sentencepiece.bpe.model: 0%| | 0.00/5.07M [00:00<?, ?B/s]tokenizer.json: 0%| | 0.00/9.10M [00:00<?, ?B/s]model.safetensors: 0%| | 0.00/1.12G [00:00<?, ?B/s]
🔥 Enhanced Oversampling: min=1.00, max=25.33
├─ Objective boosted samples: 405 (target: weak class at 40% F1)
└─ Neutral boosted samples: 1874 (target: weak class at 49% F1)
[3190/3190 2:11:53, Epoch 21/22]
Epoch Training Loss Validation Loss Sent Acc Sent Prec Sent Rec Sent F1 Pol Acc Pol Prec Pol Rec Pol F1 Macro F1 Avg
0 1.221900 No log 0.282274 0.276681 0.372570 0.216651 0.062207 0.020736 0.333333 0.039043 0.127847
1 0.958100 No log 0.450836 0.577043 0.549206 0.476630 0.173244 0.471839 0.422947 0.192639 0.334635
3 0.580100 No log 0.444147 0.676228 0.550158 0.479831 0.567224 0.518697 0.642482 0.503606 0.491719
4 0.405500 No log 0.534448 0.669483 0.611339 0.572456 0.482943 0.509565 0.603292 0.450955 0.511705
6 0.237700 No log 0.478930 0.607806 0.620587 0.516157 0.602676 0.561887 0.603275 0.542113 0.529135
7 0.154200 No log 0.563211 0.662557 0.658165 0.599737 0.524415 0.556211 0.575922 0.495231 0.547484
9 0.141300 No log 0.609365 0.645971 0.670817 0.628935 0.727759 0.607912 0.624140 0.608379 0.618657
10 0.125300 No log 0.686288 0.684739 0.673612 0.677391 0.683612 0.597016 0.591747 0.569636 0.623513
12 0.174600 No log 0.544482 0.659230 0.655613 0.586739 0.654181 0.640216 0.566274 0.542026 0.564383
13 0.112000 No log 0.613378 0.655529 0.678136 0.636948 0.749164 0.635493 0.590719 0.592557 0.614752
15 0.076900 No log 0.648829 0.666366 0.678644 0.660079 0.750502 0.640797 0.598403 0.600867 0.630473
16 0.087400 No log 0.634114 0.660302 0.682223 0.650921 0.736455 0.637921 0.592843 0.590996 0.620958
18 0.098800 No log 0.650836 0.675646 0.678335 0.663736 0.741137 0.647821 0.602180 0.601890 0.632813
19 0.076000 No log 0.657525 0.681795 0.683581 0.668998 0.720401 0.616893 0.602205 0.592564 0.630781
21 0.058500 No log 0.658863 0.679045 0.686802 0.669597 0.731104 0.625376 0.600988 0.594955 0.632276
✅ SECTION 10: Model Training Execution completed in 2.2h 14m
🕒 Total runtime so far: 2.2h 14m

---

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key base_name test_test_sent_acc test_test_sent_prec test_test_sent_rec test_test_sent_f1 test_test_pol_acc test_test_pol_prec test_test_pol_rec test_test_pol_f1 test_test_macro_f1_avg test_test_runtime test_test_samples_per_second test_test_steps_per_second
0 xlm_roberta xlm-roberta-base 0.692977 0.692564 0.711987 0.691361 0.732441 0.647557 0.620788 0.620669 0.656015 5.5197 270.85 17.03

=== Detailed breakdowns for xlm_roberta ===

Sentiment — per class (precision/recall/F1/support):
class precision recall f1 support
0 negative 0.855908 0.670429 0.751899 886
1 neutral 0.484694 0.710723 0.576340 401
2 positive 0.737089 0.754808 0.745843 208

Polarization — per class (precision/recall/F1/support):
class precision recall f1 support
0 non_polarized 0.563847 0.781609 0.655106 435
1 objective 0.508475 0.333333 0.402685 90
2 partisan 0.870348 0.747423 0.804215 970

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice support accuracy macro_f1 f1_non_polarized f1_objective f1_partisan
0 negative 886 0.775395 0.587774 0.575472 0.333333 0.854518
1 neutral 401 0.665835 0.603829 0.735683 0.465116 0.610687
2 positive 208 0.677885 0.554547 0.637500 0.285714 0.740426

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice support accuracy macro_f1 f1_negative f1_neutral f1_positive
0 partisan 970 0.701031 0.643891 0.794304 0.381643 0.755725
1 non_polarized 435 0.687356 0.693044 0.587786 0.717622 0.773723
2 objective 90 0.633333 0.573880 0.555556 0.711538 0.454545

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
✓ Optimal bias vector found (VAL macro-F1=0.302):
• non_polarized: -0.80
• objective: +0.10
• partisan: +0.30
📈 Step 3: Evaluating calibration impact on test set...

📊 TEST MACRO-F1: 0.150 → 0.275 (+0.125)

Per-class breakdown:
📉 non_polarized: P=0.291 R=1.000 F1=0.451 (n=435) → P=0.000 R=0.000 F1=0.000 (-0.451)
📈 objective: P=0.000 R=0.000 F1=0.000 (n=90) → P=0.182 R=0.022 F1=0.040 (+0.040)
📈 partisan: P=0.000 R=0.000 F1=0.000 (n=970) → P=0.649 R=0.993 F1=0.785 (+0.785)

✅ Calibration complete! Bias vector saved to:
./runs_xlm_roberta_optimized/calibration_vector/xlm_roberta_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!

[xlm_roberta] xlm-roberta-base
Token indices sequence length is longer than the specified maximum sequence length for this model (950 > 512). Running this sequence through the model will result in indexing errors
Token length stats: {'mean': 106.9514, 'p50': 96.0, 'p90': 170.0, 'p95': 182.0, 'p99': 215.0, 'max': 950}
✅ SECTION 11+: Evaluation & Calibration completed in 23.3s
🕒 Total runtime so far: 2.2h 15m

---

============================================================
⏱️ EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports : 16.0s
SECTION 3: Configuration Setup : 0.0s
SECTION 4: Data Loading & Preprocessing : 0.1s
SECTION 5-9: Model Architecture & Training Setup : 0.1s
SECTION 10: Model Training Execution : 2.2h 14m
SECTION 11+: Evaluation & Calibration : 23.3s
======================================== : ==========
TOTAL EXECUTION TIME : 2.2h 15m
============================================================
