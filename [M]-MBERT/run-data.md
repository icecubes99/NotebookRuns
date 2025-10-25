🚀 Starting SECTION 10: Model Training Execution...

=== Running mbert -> bert-base-multilingual-cased ===
🔥 Enhanced Oversampling: min=1.00, max=68.28
   ├─ Objective boosted samples: 405 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 1874 (target: weak class at 49% F1)
 [2900/2900 1:04:34, Epoch 19/20]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	0.954500	No log	0.294314	0.279168	0.404680	0.257689	0.070903	0.227380	0.344332	0.060450	0.159069
1	0.720200	No log	0.385284	0.611575	0.509928	0.408897	0.204682	0.532474	0.425929	0.203238	0.306067
3	0.317800	No log	0.410702	0.658244	0.502510	0.427790	0.319064	0.477876	0.533772	0.308423	0.368107
4	0.213500	No log	0.476923	0.682413	0.567313	0.514464	0.450167	0.504774	0.595333	0.437549	0.476007
6	0.156700	No log	0.468896	0.682777	0.560002	0.504724	0.671572	0.574512	0.626160	0.583988	0.544356
7	0.142200	No log	0.477592	0.639334	0.617709	0.518202	0.555184	0.556968	0.599348	0.518259	0.518231
9	0.122400	No log	0.461538	0.660749	0.581123	0.505166	0.565217	0.572649	0.575026	0.518647	0.511907
10	0.109600	No log	0.517726	0.599892	0.635082	0.543422	0.714381	0.596422	0.598651	0.592163	0.567793
12	0.081800	No log	0.584615	0.664324	0.634730	0.611982	0.713043	0.594331	0.616834	0.600550	0.606266
13	0.073000	No log	0.564548	0.644490	0.658293	0.598173	0.630769	0.582813	0.581729	0.547570	0.572872
15	0.071800	No log	0.580602	0.659895	0.655638	0.613011	0.620736	0.581251	0.569069	0.534751	0.573881
16	0.067100	No log	0.555853	0.654030	0.646576	0.594497	0.603344	0.580198	0.565978	0.521709	0.558103
18	0.083000	No log	0.591304	0.656027	0.666797	0.621141	0.661538	0.581634	0.566628	0.541397	0.581269
19	0.073200	No log	0.591304	0.653566	0.666344	0.620220	0.672910	0.581231	0.569480	0.551774	0.585997
✅ SECTION 10: Model Training Execution completed in 1.1h 5m
🕒 Total runtime so far: 2.5h 29m
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	mbert	bert-base-multilingual-cased	0.606689	0.678229	0.654576	0.632456	0.713043	0.604058	0.620417	0.607349	0.619903	5.2075	287.085	18.051

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

