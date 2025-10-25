🚀 Starting SECTION 10: Model Training Execution...

=== Running mbert -> bert-base-multilingual-cased ===
🔥 Enhanced Oversampling: min=1.00, max=68.28
   ├─ Objective boosted samples: 405 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 1874 (target: weak class at 49% F1)
 [2900/2900 1:28:50, Epoch 19/20]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	0.978500	No log	0.286288	0.295020	0.384575	0.235865	0.063545	0.087542	0.335025	0.042591	0.139228
1	0.686600	No log	0.408027	0.568599	0.535749	0.429385	0.131104	0.505129	0.380793	0.129050	0.279218
3	0.348200	No log	0.437458	0.610303	0.556629	0.469762	0.408696	0.493009	0.586622	0.390636	0.430199
4	0.201200	No log	0.442809	0.640311	0.575026	0.487517	0.530435	0.519211	0.620263	0.495638	0.491578
6	0.161100	No log	0.458194	0.654705	0.586600	0.507099	0.687625	0.570976	0.625689	0.585711	0.546405
7	0.148300	No log	0.514381	0.637333	0.636090	0.554051	0.523077	0.540977	0.544786	0.471985	0.513018
9	0.144300	No log	0.504348	0.659596	0.623994	0.548298	0.628763	0.565814	0.614626	0.558185	0.553241
10	0.131200	No log	0.517726	0.594153	0.636723	0.542526	0.678930	0.584251	0.591432	0.573906	0.558216
12	0.089100	No log	0.593311	0.658854	0.645338	0.617320	0.729766	0.596295	0.607773	0.600740	0.609030
13	0.077100	No log	0.557860	0.651986	0.650172	0.594662	0.646823	0.584169	0.585381	0.553721	0.574192
15	0.075400	No log	0.584615	0.633675	0.651508	0.608432	0.652843	0.574221	0.584234	0.553016	0.580724
16	0.070300	No log	0.527090	0.644701	0.627398	0.566942	0.572575	0.580473	0.545129	0.495882	0.531412
18	0.084200	No log	0.598662	0.658336	0.658220	0.624428	0.653512	0.575071	0.570005	0.545385	0.584907
19	0.071500	No log	0.585953	0.653069	0.655680	0.615795	0.659532	0.577783	0.570920	0.549015	0.582405
✅ SECTION 10: Model Training Execution completed in 1.5h 29m
🕒 Total runtime so far: 6.2h 14m
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	mbert	bert-base-multilingual-cased	0.610702	0.657814	0.656417	0.627949	0.738462	0.626185	0.642668	0.633175	0.630562	4.5973	325.193	20.447

=== Detailed breakdowns for mbert ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.833631	0.525959	0.644983	886
1	neutral	0.411606	0.760599	0.534151	401
2	positive	0.728205	0.682692	0.704715	208

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.602434	0.682759	0.640086	435
1	objective	0.431579	0.455556	0.443243	90
2	partisan	0.844542	0.789691	0.816196	970

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.792325	0.591719	0.576177	0.327869	0.871111
1	neutral	401	0.635910	0.607597	0.681265	0.536082	0.605442
2	positive	208	0.706731	0.604602	0.679487	0.370370	0.763948

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	970	0.609278	0.585173	0.704565	0.355641	0.695312
1	non_polarized	435	0.606897	0.597686	0.362832	0.678227	0.752000
2	objective	90	0.644444	0.605726	0.551724	0.720000	0.545455

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
   ✓ Optimal bias vector found (VAL macro-F1=0.606):
      • non_polarized: -0.10
      •     objective: -0.80
      •      partisan: +0.00
📈 Step 3: Evaluating calibration impact on test set...

   📊 TEST MACRO-F1: 0.633 → 0.633 (+0.001)

   Per-class breakdown:
   📉 non_polarized: P=0.601 R=0.683 F1=0.639 (n=435)  →  P=0.657 R=0.591 F1=0.622 (-0.017)
   📉     objective: P=0.432 R=0.456 F1=0.443 (n=90)  →  P=0.480 R=0.400 F1=0.436 (-0.007)
   📈      partisan: P=0.844 R=0.789 F1=0.816 (n=970)  →  P=0.817 R=0.867 F1=0.841 (+0.026)

✅ Calibration complete! Bias vector saved to:
   ./runs_mbert_optimized/calibration_vector/mbert_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!

[mbert] bert-base-multilingual-cased
Token indices sequence length is longer than the specified maximum sequence length for this model (916 > 512). Running this sequence through the model will result in indexing errors
Token length stats: {'mean': 109.174, 'p50': 97.0, 'p90': 179.0, 'p95': 194.0, 'p99': 226.02000000000044, 'max': 916}
✅ SECTION 11+: Evaluation & Calibration completed in 6.3m 16s
🕒 Total runtime so far: 6.3h 20m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 9.2s
SECTION 3: Configuration Setup           : 24.4m 26s
SECTION 4: Data Loading & Preprocessing  : 1.2m 11s
SECTION 5-9: Model Architecture & Training Setup : 29.3s
SECTION 10: Model Training Execution     : 1.5h 29m
SECTION 11+: Evaluation & Calibration    : 6.3m 16s
======================================== : ==========
TOTAL EXECUTION TIME                     : 6.3h 20m
============================================================