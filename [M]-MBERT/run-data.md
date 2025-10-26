🚀 Starting SECTION 10: Model Training Execution...

=== Running mbert -> bert-base-multilingual-cased ===
🔥 Enhanced Oversampling: min=1.00, max=68.28
   ├─ Objective boosted samples: 405 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 1874 (target: weak class at 49% F1)
 [ 146/2900 01:38 < 31:32, 1.46 it/s, Epoch 1.00/20]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	3.133500	No log	0.269565	0.423025	0.334944	0.144555	0.062207	0.020736	0.333333	0.039043	0.091799
 [2900/2900 1:39:47, Epoch 19/20]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	3.133500	No log	0.269565	0.423025	0.334944	0.144555	0.062207	0.020736	0.333333	0.039043	0.091799
2	1.134300	No log	0.301672	0.273150	0.427080	0.280311	0.125753	0.411888	0.368195	0.112268	0.196290
4	0.379000	No log	0.387291	0.624080	0.553922	0.416336	0.255518	0.480124	0.527970	0.243526	0.329931
6	0.235800	No log	0.360535	0.640953	0.538341	0.398770	0.333779	0.518705	0.550354	0.348906	0.373838
8	0.166200	No log	0.423411	0.632283	0.584314	0.470431	0.354515	0.549426	0.507033	0.368667	0.419549
10	0.190800	No log	0.442809	0.618814	0.602504	0.480105	0.488294	0.547273	0.547273	0.467705	0.473905
12	0.110000	No log	0.545819	0.609368	0.648119	0.576113	0.618060	0.554248	0.587529	0.549158	0.562636
14	0.121000	No log	0.494983	0.614916	0.621884	0.536590	0.507023	0.565569	0.505184	0.447804	0.492197
16	0.099100	No log	0.523077	0.635105	0.634436	0.565092	0.587960	0.576543	0.545489	0.513599	0.539346
18	0.101300	No log	0.520401	0.631692	0.632353	0.563244	0.606020	0.582323	0.551679	0.522322	0.542783
19	0.087100	No log	0.518395	0.631326	0.631802	0.560446	0.614047	0.580917	0.557001	0.531494	0.545970
✅ SECTION 10: Model Training Execution completed in 1.7h 41m
🕒 Total runtime so far: 5.0h 59m
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...

=== Detailed breakdowns for mbert ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.861650	0.400677	0.546995	886
1	neutral	0.394541	0.793017	0.526926	401
2	positive	0.592058	0.788462	0.676289	208

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.454031	0.737931	0.562172	435
1	objective	0.316832	0.355556	0.335079	90
2	partisan	0.834061	0.590722	0.691611	970

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.676072	0.487769	0.457778	0.225352	0.780176
1	neutral	401	0.578554	0.508907	0.678937	0.428571	0.419214
2	positive	208	0.456731	0.421243	0.512315	0.333333	0.418079

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	970	0.536082	0.537635	0.604915	0.332750	0.675241
1	non_polarized	435	0.581609	0.527456	0.191489	0.684211	0.706667
2	objective	90	0.711111	0.647436	0.653846	0.788462	0.500000

Saved detailed breakdowns to: ./runs_mbert_optimized/details

🎯 MULTICLASS CALIBRATION - Optimize prediction biases for better performance
======================================================================

🔧 Calibrating mbert (bert-base-multilingual-cased)...
📊 Step 1: Extracting polarization logits from trained model...
   Loading model from: ./runs_mbert_optimized/mbert
   ✓ Loading weights from: ./runs_mbert_optimized/mbert/model.safetensors
   Loading model from: ./runs_mbert_optimized/mbert
   ✓ Loading weights from: ./runs_mbert_optimized/mbert/model.safetensors
   ✓ Validation logits shape: (1495, 3)
   ✓ Test logits shape: (1495, 3)
🔍 Step 2: Searching for optimal bias vector (coordinate search)...
   ✓ Optimal bias vector found (VAL macro-F1=0.574):
      • non_polarized: -0.20
      •     objective: +0.40
      •      partisan: +0.00
📈 Step 3: Evaluating calibration impact on test set...

   📊 TEST MACRO-F1: 0.530 → 0.572 (+0.041)

   Per-class breakdown:
   📉 non_polarized: P=0.455 R=0.738 F1=0.563 (n=435)  →  P=0.610 R=0.510 F1=0.556 (-0.007)
   📈     objective: P=0.317 R=0.356 F1=0.335 (n=90)  →  P=0.321 R=0.389 F1=0.352 (+0.017)
   📈      partisan: P=0.835 R=0.593 F1=0.693 (n=970)  →  P=0.787 R=0.829 F1=0.807 (+0.114)

✅ Calibration complete! Bias vector saved to:
   ./runs_mbert_optimized/calibration_vector/mbert_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!


[mbert] bert-base-multilingual-cased
Token indices sequence length is longer than the specified maximum sequence length for this model (916 > 512). Running this sequence through the model will result in indexing errors
Token length stats: {'mean': 109.174, 'p50': 97.0, 'p90': 179.0, 'p95': 194.0, 'p99': 226.02000000000044, 'max': 916}
✅ SECTION 11+: Evaluation & Calibration completed in 28.5m 30s
🕒 Total runtime so far: 5.5h 27m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 8.9s
SECTION 3: Configuration Setup           : 1.7h 40m
SECTION 4: Data Loading & Preprocessing  : 14.0s
SECTION 5-9: Model Architecture & Training Setup : 1.4m 23s
SECTION 10: Model Training Execution     : 1.7h 41m
SECTION 11+: Evaluation & Calibration    : 28.5m 30s
======================================== : ==========
TOTAL EXECUTION TIME                     : 5.5h 27m
============================================================