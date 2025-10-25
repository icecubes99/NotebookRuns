🚀 Starting SECTION 10: Model Training Execution...

=== Running mbert -> bert-base-multilingual-cased ===
tokenizer_config.json:   0%|          | 0.00/49.0 [00:00<?, ?B/s]config.json:   0%|          | 0.00/625 [00:00<?, ?B/s]vocab.txt:   0%|          | 0.00/996k [00:00<?, ?B/s]tokenizer.json:   0%|          | 0.00/1.96M [00:00<?, ?B/s]model.safetensors:   0%|          | 0.00/714M [00:00<?, ?B/s]
🔥 Enhanced Oversampling: min=1.00, max=68.28
   ├─ Objective boosted samples: 405 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 1874 (target: weak class at 49% F1)
 [2616/2900 57:15 < 06:13, 0.76 it/s, Epoch 18/20]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	1.143400	No log	0.268896	0.089632	0.333333	0.141276	0.062207	0.020736	0.333333	0.039043	0.090159
1	0.948000	No log	0.269565	0.423025	0.334944	0.144555	0.062207	0.020736	0.333333	0.039043	0.091799
3	0.677500	No log	0.346488	0.585167	0.494278	0.357877	0.172575	0.468766	0.422970	0.193428	0.275653
4	0.528400	No log	0.513712	0.644447	0.524632	0.509758	0.486288	0.507020	0.575548	0.419738	0.464748
6	0.356400	No log	0.401338	0.680054	0.512302	0.427373	0.527090	0.504729	0.623186	0.478751	0.453062
7	0.388100	No log	0.408696	0.658292	0.504303	0.427709	0.544482	0.502889	0.629015	0.489064	0.458386
9	0.276100	No log	0.442809	0.632422	0.563840	0.476960	0.418729	0.490286	0.589005	0.399033	0.437997
10	0.282300	No log	0.538462	0.565254	0.613036	0.542352	0.609365	0.527025	0.655939	0.534619	0.538486
12	0.210000	No log	0.429431	0.667329	0.533571	0.461921	0.511706	0.510306	0.622634	0.474000	0.467960
13	0.224500	No log	0.440134	0.661168	0.556868	0.478901	0.529097	0.525066	0.628009	0.497606	0.488253
15	0.269800	No log	0.432776	0.664303	0.548121	0.472640	0.520401	0.518116	0.633117	0.484652	0.478646
16	0.201300	No log	0.415385	0.676288	0.535215	0.454500	0.458194	0.513173	0.586480	0.443648	0.449074
18	0.236600	No log	0.459532	0.657649	0.568638	0.500680	0.553846	0.529415	0.643821	0.511725	0.506202
✅ SECTION 10: Model Training Execution completed in 57.8m 46s
🕒 Total runtime so far: 60.0m 59s
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	mbert	bert-base-multilingual-cased	0.53913	0.555329	0.605001	0.535048	0.612709	0.536327	0.654056	0.53849	0.536769	4.886	305.979	19.239

=== Detailed breakdowns for mbert ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.850325	0.442438	0.582034	886
1	neutral	0.385838	0.665835	0.488564	401
2	positive	0.429825	0.706731	0.534545	208

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.527675	0.657471	0.585466	435
1	objective	0.217391	0.722222	0.334190	90
2	partisan	0.863914	0.582474	0.695813	970

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.648984	0.487851	0.525253	0.176796	0.761506
1	neutral	401	0.581047	0.555678	0.655172	0.509804	0.502058
2	positive	208	0.519231	0.483506	0.560000	0.363636	0.526882

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	970	0.549485	0.508176	0.655797	0.318271	0.550459
1	non_polarized	435	0.491954	0.441958	0.199005	0.604255	0.522613
2	objective	90	0.655556	0.554929	0.476190	0.771930	0.416667

Saved detailed breakdowns to: ./runs_mbert_optimized/details

🎯 MULTICLASS CALIBRATION - Optimize prediction biases for better performance
======================================================================

🔧 Calibrating mbert (bert-base-multilingual-cased)...
📊 Step 1: Extracting polarization logits from trained model...
   Loading model from: ./runs_mbert_optimized/mbert
   Warning: No trained weights found at ./runs_mbert_optimized/mbert/pytorch_model.bin, using untrained model
   Loading model from: ./runs_mbert_optimized/mbert
   Warning: No trained weights found at ./runs_mbert_optimized/mbert/pytorch_model.bin, using untrained model
   ✓ Validation logits shape: (1495, 3)
   ✓ Test logits shape: (1495, 3)
🔍 Step 2: Searching for optimal bias vector (coordinate search)...
   ✓ Optimal bias vector found (VAL macro-F1=0.358):
      • non_polarized: -0.30
      •     objective: +0.20
      •      partisan: +0.00
📈 Step 3: Evaluating calibration impact on test set...

   📊 TEST MACRO-F1: 0.292 → 0.285 (-0.007)

   Per-class breakdown:
   📉 non_polarized: P=0.273 R=0.007 F1=0.013 (n=435)  →  P=0.000 R=0.000 F1=0.000 (-0.013)
   📈     objective: P=0.095 R=0.133 F1=0.111 (n=90)  →  P=0.106 R=0.511 F1=0.176 (+0.065)
   📉      partisan: P=0.644 R=0.901 F1=0.751 (n=970)  →  P=0.649 R=0.710 F1=0.678 (-0.073)

✅ Calibration complete! Bias vector saved to:
   ./runs_mbert_optimized/calibration_vector/mbert_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!

[mbert] bert-base-multilingual-cased
Token indices sequence length is longer than the specified maximum sequence length for this model (916 > 512). Running this sequence through the model will result in indexing errors
Token length stats: {'mean': 109.174, 'p50': 97.0, 'p90': 179.0, 'p95': 194.0, 'p99': 226.02000000000044, 'max': 916}
✅ SECTION 11+: Evaluation & Calibration completed in 16.5s
🕒 Total runtime so far: 1.0h 0m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 9.2s
SECTION 3: Configuration Setup           : 13.1s
SECTION 4: Data Loading & Preprocessing  : 1.2m 11s
SECTION 5-9: Model Architecture & Training Setup : 33.7s
SECTION 10: Model Training Execution     : 57.8m 46s
SECTION 11+: Evaluation & Calibration    : 16.5s
======================================== : ==========
TOTAL EXECUTION TIME                     : 1.0h 0m
============================================================