# SEED 43 - RUN 11

🚀 Starting SECTION 10: Model Training Execution...

=== Running mbert -> bert-base-multilingual-cased ===
🔥 Enhanced Oversampling: min=1.00, max=68.28
   ├─ Objective boosted samples: 405 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 1874 (target: weak class at 49% F1)
 [2900/2900 1:27:38, Epoch 19/20]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	0.951100	No log	0.289632	0.277227	0.393407	0.245212	0.062207	0.020736	0.333333	0.039043	0.142127
1	0.695600	No log	0.359197	0.614755	0.486210	0.373452	0.151839	0.593670	0.386922	0.139682	0.256567
3	0.331600	No log	0.406020	0.619189	0.521184	0.426333	0.343144	0.482140	0.552861	0.331405	0.378869
4	0.209100	No log	0.477592	0.655548	0.564518	0.509287	0.549833	0.518808	0.638229	0.504101	0.506694
6	0.171000	No log	0.474247	0.670811	0.565808	0.509119	0.607358	0.542973	0.605044	0.540341	0.524730
7	0.156200	No log	0.490970	0.612563	0.607082	0.523477	0.541137	0.539451	0.583481	0.500925	0.512201
9	0.119700	No log	0.482274	0.667000	0.583898	0.522515	0.632107	0.569123	0.608034	0.563131	0.542823
10	0.127200	No log	0.546488	0.612787	0.648214	0.571413	0.711706	0.583721	0.600713	0.590544	0.580979
12	0.091900	No log	0.550502	0.669338	0.643940	0.593605	0.656187	0.584811	0.609309	0.578291	0.585948
13	0.077800	No log	0.563211	0.635846	0.650090	0.593950	0.641472	0.587016	0.570721	0.549575	0.571762
15	0.071600	No log	0.595987	0.652145	0.666182	0.619655	0.612709	0.575121	0.559947	0.528374	0.574014
16	0.072600	No log	0.546488	0.646221	0.641886	0.581800	0.569231	0.580474	0.533522	0.492211	0.537005
18	0.069000	No log	0.590635	0.664942	0.653627	0.619361	0.639465	0.585657	0.554846	0.532820	0.576091
19	0.066700	No log	0.588629	0.665081	0.656903	0.617263	0.666890	0.583357	0.563412	0.548489	0.582876
✅ SECTION 10: Model Training Execution completed in 1.5h 29m
🕒 Total runtime so far: 8.1h 4m
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	mbert	bert-base-multilingual-cased	0.555853	0.656906	0.63631	0.589911	0.664883	0.585905	0.60586	0.575785	0.582848	5.0878	293.84	18.476

=== Detailed breakdowns for mbert ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.858173	0.402935	0.548387	886
1	neutral	0.375703	0.832918	0.517829	401
2	positive	0.736842	0.673077	0.703518	208

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.495810	0.816092	0.616855	435
1	objective	0.386364	0.377778	0.382022	90
2	partisan	0.875543	0.623711	0.728477	970

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.685102	0.515972	0.500000	0.266667	0.781250
1	neutral	401	0.638404	0.575057	0.726891	0.477273	0.521008
2	positive	208	0.629808	0.546128	0.648045	0.333333	0.657005

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	970	0.530928	0.551454	0.603810	0.339089	0.711462
1	non_polarized	435	0.583908	0.543643	0.243655	0.677596	0.709677
2	objective	90	0.688889	0.640826	0.581818	0.769231	0.571429

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
   ✓ Optimal bias vector found (VAL macro-F1=0.623):
      • non_polarized: -0.30
      •     objective: +0.10
      •      partisan: +0.00
📈 Step 3: Evaluating calibration impact on test set...

   📊 TEST MACRO-F1: 0.577 → 0.609 (+0.032)

   Per-class breakdown:
   📉 non_polarized: P=0.497 R=0.818 F1=0.619 (n=435)  →  P=0.659 R=0.577 F1=0.615 (-0.003)
   📈     objective: P=0.386 R=0.378 F1=0.382 (n=90)  →  P=0.380 R=0.389 F1=0.385 (+0.003)
   📈      partisan: P=0.877 R=0.625 F1=0.730 (n=970)  →  P=0.806 R=0.849 F1=0.827 (+0.098)

✅ Calibration complete! Bias vector saved to:
   ./runs_mbert_optimized/calibration_vector/mbert_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!

[mbert] bert-base-multilingual-cased
Token indices sequence length is longer than the specified maximum sequence length for this model (916 > 512). Running this sequence through the model will result in indexing errors
Token length stats: {'mean': 109.174, 'p50': 97.0, 'p90': 179.0, 'p95': 194.0, 'p99': 226.02000000000044, 'max': 916}
✅ SECTION 11+: Evaluation & Calibration completed in 3.7m 41s
🕒 Total runtime so far: 8.1h 8m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 9.2s
SECTION 3: Configuration Setup           : 21.5m 30s
SECTION 4: Data Loading & Preprocessing  : 1.2m 11s
SECTION 5-9: Model Architecture & Training Setup : 5.8s
SECTION 10: Model Training Execution     : 1.5h 29m
SECTION 11+: Evaluation & Calibration    : 3.7m 41s
======================================== : ==========
TOTAL EXECUTION TIME                     : 8.1h 8m
============================================================