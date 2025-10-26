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

# SEED 44 - RUN 12

🚀 Starting SECTION 10: Model Training Execution...

=== Running mbert -> bert-base-multilingual-cased ===
🔥 Enhanced Oversampling: min=1.00, max=68.28
   ├─ Objective boosted samples: 405 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 1874 (target: weak class at 49% F1)
 [2900/2900 1:09:56, Epoch 19/20]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	0.969300	No log	0.290301	0.292898	0.395799	0.250409	0.062207	0.020736	0.333333	0.039043	0.144726
1	0.710200	No log	0.411371	0.629624	0.527647	0.438483	0.127090	0.514514	0.376232	0.121295	0.279889
3	0.319100	No log	0.438127	0.594566	0.546352	0.458398	0.333779	0.490500	0.558507	0.321603	0.390001
4	0.214500	No log	0.527759	0.658857	0.566103	0.548146	0.507692	0.515384	0.621873	0.481161	0.514653
6	0.162200	No log	0.446154	0.658269	0.576172	0.488032	0.648829	0.556537	0.624000	0.564950	0.526491
7	0.136100	No log	0.529097	0.617152	0.629183	0.558428	0.592642	0.551954	0.578278	0.524904	0.541666
9	0.112800	No log	0.503010	0.659670	0.607120	0.540867	0.634114	0.558533	0.574814	0.536330	0.538598
10	0.110800	No log	0.493645	0.583895	0.633879	0.511696	0.680936	0.577974	0.579248	0.563235	0.537465
12	0.079000	No log	0.609365	0.632713	0.658883	0.622734	0.703010	0.568739	0.605367	0.580283	0.601509
13	0.073600	No log	0.577926	0.634636	0.660507	0.604148	0.650836	0.576140	0.579312	0.549236	0.576692
15	0.073400	No log	0.569231	0.623668	0.661583	0.592306	0.622742	0.564593	0.564231	0.528330	0.560318
16	0.071500	No log	0.575251	0.644953	0.660735	0.603045	0.595987	0.571767	0.554288	0.510727	0.556886
18	0.086800	No log	0.593311	0.652943	0.663771	0.620535	0.692977	0.600369	0.583170	0.569474	0.595004
19	0.069800	No log	0.584615	0.653479	0.663160	0.616182	0.688294	0.594060	0.580855	0.565373	0.590778
✅ SECTION 10: Model Training Execution completed in 1.2h 10m
🕒 Total runtime so far: 9.3h 20m
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...

=== Detailed breakdowns for mbert ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.813149	0.530474	0.642077	886
1	neutral	0.420749	0.728180	0.533333	401
2	positive	0.677130	0.725962	0.700696	208

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.556738	0.721839	0.628629	435
1	objective	0.373832	0.444444	0.406091	90
2	partisan	0.847087	0.719588	0.778149	970

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.749436	0.546646	0.552764	0.250000	0.837174
1	neutral	401	0.635910	0.599406	0.697674	0.520833	0.579710
2	positive	208	0.639423	0.573766	0.631579	0.413793	0.675926

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	970	0.611340	0.579641	0.703578	0.346939	0.688406
1	non_polarized	435	0.604598	0.598440	0.360515	0.677228	0.757576
2	objective	90	0.633333	0.589352	0.526316	0.720000	0.521739

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
   ✓ Optimal bias vector found (VAL macro-F1=0.599):
      • non_polarized: -0.20
      •     objective: +0.70
      •      partisan: +0.00
📈 Step 3: Evaluating calibration impact on test set...

   📊 TEST MACRO-F1: 0.604 → 0.615 (+0.011)

   Per-class breakdown:
   📉 non_polarized: P=0.557 R=0.722 F1=0.629 (n=435)  →  P=0.667 R=0.584 F1=0.623 (-0.006)
   📉     objective: P=0.374 R=0.444 F1=0.406 (n=90)  →  P=0.353 R=0.467 F1=0.402 (-0.004)
   📈      partisan: P=0.847 R=0.720 F1=0.778 (n=970)  →  P=0.810 R=0.831 F1=0.820 (+0.042)

✅ Calibration complete! Bias vector saved to:
   ./runs_mbert_optimized/calibration_vector/mbert_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!
[mbert] bert-base-multilingual-cased
Token indices sequence length is longer than the specified maximum sequence length for this model (916 > 512). Running this sequence through the model will result in indexing errors
Token length stats: {'mean': 109.174, 'p50': 97.0, 'p90': 179.0, 'p95': 194.0, 'p99': 226.02000000000044, 'max': 916}
✅ SECTION 11+: Evaluation & Calibration completed in 44.8s
🕒 Total runtime so far: 9.3h 20m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 9.2s
SECTION 3: Configuration Setup           : 5.3m 18s
SECTION 4: Data Loading & Preprocessing  : 1.2m 11s
SECTION 5-9: Model Architecture & Training Setup : 5.3s
SECTION 10: Model Training Execution     : 1.2h 10m
SECTION 11+: Evaluation & Calibration    : 44.8s
======================================== : ==========
TOTAL EXECUTION TIME                     : 9.3h 20m
============================================================

# SEED 45 - RUN 12
================================================================================
🎯 mBERT TRAINING - RUN #11: SEED ENSEMBLE STRATEGY (SEED = 45)
================================================================================
📊 Run History: R1→R8: 58.5%-61.99% → R9: 62.74% → R10: 63.06% 🏆 NEW BEST!
   Run #9 Result:  62.74% Macro-F1 (+0.68% vs R4)
   Run #10 Result: 63.06% Macro-F1 (+0.32% vs R9, +1.00% vs R4)
   SEED=42 Baseline: 62.90% (average of R9 + R10)
   Key Finding: R10 validated R9 stability - configuration is reproducible!
   Run #11-13 Strategy: SEED ENSEMBLE (train 3 more models with seeds 43, 44, 45)

🎲 Current Seed: 45
   ├─ Run #9:  SEED = 42 ✅ COMPLETED (62.74%)
   ├─ Run #10: SEED = 42 ✅ COMPLETED (63.06%)
   ├─ Run #11: SEED = 43 ← CURRENT RUN (Expected: 62-63%)
   ├─ Run #12: SEED = 44 (Expected: 62-63%)
   └─ Run #13: SEED = 45 (Expected: 62-63%)

🎯 Expected Outcome for Run #11:
   ├─ Individual model: 62-63% Macro-F1 (consistent with R9/R10)
   ├─ After Run #13: 4 models trained (seeds 42, 43, 44, 45)
   └─ Final Ensemble: 64-65% Macro-F1 (+1-2% from diversity)

✅ Bug Fixes Applied (from R9, validated in R10):
   ├─ transformers.set_seed(45) - eliminates per-class variance
   └─ Robust calibration loading - checks multiple weight file formats
...
🕒 Total runtime so far: 1.9m 53s
------------------------------------------------------------

🚀 Starting SECTION 4: Data Loading & Preprocessing...
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...


🚀 Starting SECTION 10: Model Training Execution...

=== Running mbert -> bert-base-multilingual-cased ===
tokenizer_config.json:   0%|          | 0.00/49.0 [00:00<?, ?B/s]config.json:   0%|          | 0.00/625 [00:00<?, ?B/s]vocab.txt:   0%|          | 0.00/996k [00:00<?, ?B/s]tokenizer.json:   0%|          | 0.00/1.96M [00:00<?, ?B/s]model.safetensors:   0%|          | 0.00/714M [00:00<?, ?B/s]
🔥 Enhanced Oversampling: min=1.00, max=68.28
   ├─ Objective boosted samples: 405 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 1874 (target: weak class at 49% F1)
 [2900/2900 1:30:22, Epoch 19/20]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	0.988500	No log	0.284281	0.306265	0.379744	0.230474	0.062207	0.020736	0.333333	0.039043	0.134758
1	0.737900	No log	0.377258	0.619323	0.515411	0.401210	0.125753	0.529478	0.371448	0.112560	0.256885
3	0.318400	No log	0.438796	0.607465	0.547962	0.460761	0.367893	0.475620	0.564550	0.351854	0.406308
4	0.202900	No log	0.466890	0.690591	0.555171	0.503897	0.401338	0.499081	0.574124	0.393086	0.448492
6	0.170200	No log	0.448829	0.656366	0.573600	0.498235	0.640803	0.553104	0.625155	0.561938	0.530087
7	0.144200	No log	0.548495	0.633326	0.639641	0.577931	0.622742	0.558249	0.622248	0.556293	0.567112
9	0.110500	No log	0.490970	0.670971	0.599895	0.535695	0.617391	0.562665	0.598728	0.548376	0.542036
10	0.108700	No log	0.546488	0.609542	0.656275	0.568106	0.693645	0.582099	0.585500	0.574606	0.571356
12	0.073500	No log	0.597324	0.674978	0.642909	0.621511	0.731104	0.599188	0.614264	0.605085	0.613298
13	0.072200	No log	0.599331	0.651549	0.667688	0.625135	0.672910	0.586218	0.597732	0.573020	0.599078
15	0.074300	No log	0.599331	0.635837	0.656785	0.615503	0.618060	0.585637	0.574415	0.541742	0.578623
16	0.062800	No log	0.551839	0.645118	0.642553	0.583953	0.603344	0.596275	0.562209	0.523780	0.553866
18	0.064100	No log	0.614047	0.657852	0.664609	0.633595	0.682943	0.603817	0.585070	0.568473	0.601034
19	0.068800	No log	0.600669	0.661436	0.660380	0.626478	0.689632	0.603374	0.589054	0.574053	0.600265
✅ SECTION 10: Model Training Execution completed in 1.5h 32m
🕒 Total runtime so far: 1.6h 34m
------------------------------------------------------------
🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	mbert	bert-base-multilingual-cased	0.597324	0.674208	0.646678	0.622599	0.735786	0.619358	0.635842	0.626347	0.624473	4.3943	340.212	21.391


=== Detailed breakdowns for mbert ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.826742	0.495485	0.619619	886
1	neutral	0.401763	0.795511	0.533891	401
2	positive	0.794118	0.649038	0.714286	208

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.608519	0.689655	0.646552	435
1	objective	0.410526	0.433333	0.421622	90
2	partisan	0.839030	0.784536	0.810868	970

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.788939	0.579164	0.584022	0.285714	0.867756
1	neutral	401	0.633416	0.593482	0.700240	0.500000	0.580205
2	positive	208	0.706731	0.629475	0.648649	0.466667	0.773109

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	970	0.585567	0.575172	0.676652	0.359788	0.689076
1	non_polarized	435	0.616092	0.613757	0.355556	0.685714	0.800000
2	objective	90	0.633333	0.581587	0.526316	0.718447	0.500000

Saved detailed breakdowns to: ./runs_mbert_optimized

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
   ✓ Optimal bias vector found (VAL macro-F1=0.608):
      • non_polarized: -0.10
      •     objective: -0.50
      •      partisan: +0.00
📈 Step 3: Evaluating calibration impact on test set...

   📊 TEST MACRO-F1: 0.626 → 0.629 (+0.003)

   Per-class breakdown:
   📉 non_polarized: P=0.609 R=0.690 F1=0.647 (n=435)  →  P=0.645 R=0.618 F1=0.631 (-0.015)
   📈     objective: P=0.411 R=0.433 F1=0.422 (n=90)  →  P=0.442 R=0.422 F1=0.432 (+0.010)
   📈      partisan: P=0.839 R=0.785 F1=0.811 (n=970)  →  P=0.816 R=0.834 F1=0.825 (+0.014)

✅ Calibration complete! Bias vector saved to:
   ./runs_mbert_optimized/calibration_vector/mbert_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!

[mbert] bert-base-multilingual-cased
Token indices sequence length is longer than the specified maximum sequence length for this model (916 > 512). Running this sequence through the model will result in indexing errors
Token length stats: {'mean': 109.174, 'p50': 97.0, 'p90': 179.0, 'p95': 194.0, 'p99': 226.02000000000044, 'max': 916}
✅ SECTION 11+: Evaluation & Calibration completed in 17.0s
🕒 Total runtime so far: 1.6h 34m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 8.9s
SECTION 3: Configuration Setup           : 1.7m 44s
SECTION 4: Data Loading & Preprocessing  : 14.0s
SECTION 5-9: Model Architecture & Training Setup : 7.9s
SECTION 10: Model Training Execution     : 1.5h 32m
SECTION 11+: Evaluation & Calibration    : 17.0s
======================================== : ==========
TOTAL EXECUTION TIME                     : 1.6h 34m
============================================================