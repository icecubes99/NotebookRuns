🚀 Starting SECTION 10: Model Training Execution...

=== Running xlm_roberta -> xlm-roberta-base ===
🔥 Enhanced Oversampling: min=0.30, max=14.88
   ├─ Objective boosted samples: 405 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 227 (target: weak class at 49% F1)
 [ 789/2900 21:34 < 57:52, 0.61 it/s, Epoch 5.42/20]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	1.155700	No log	0.318395	0.298385	0.414792	0.245746	0.100334	0.184468	0.373342	0.118325	0.182035
1	0.860200	No log	0.602676	0.554146	0.623681	0.570794	0.311037	0.493053	0.478924	0.301684	0.436239
3	0.464900	No log	0.561873	0.578177	0.582007	0.482650	0.604682	0.529379	0.651755	0.530150	0.506400
4	0.320600	No log	0.569900	0.574312	0.652512	0.566116	0.418729	0.502825	0.585590	0.409096	0.487606
 [2616/2900 1:18:53 < 08:34, 0.55 it/s, Epoch 18/20]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	1.155700	No log	0.318395	0.298385	0.414792	0.245746	0.100334	0.184468	0.373342	0.118325	0.182035
1	0.860200	No log	0.602676	0.554146	0.623681	0.570794	0.311037	0.493053	0.478924	0.301684	0.436239
3	0.464900	No log	0.561873	0.578177	0.582007	0.482650	0.604682	0.529379	0.651755	0.530150	0.506400
4	0.320600	No log	0.569900	0.574312	0.652512	0.566116	0.418729	0.502825	0.585590	0.409096	0.487606
6	0.194100	No log	0.616722	0.594119	0.674646	0.602894	0.701672	0.579205	0.635664	0.598516	0.600705
7	0.203200	No log	0.629431	0.631387	0.690821	0.632367	0.639465	0.584480	0.609995	0.564282	0.598325
9	0.138800	No log	0.707023	0.685858	0.690336	0.687578	0.761204	0.620513	0.636844	0.628070	0.657824
10	0.113600	No log	0.713043	0.667288	0.664515	0.647851	0.745151	0.618747	0.629292	0.620353	0.634102
12	0.091300	No log	0.705017	0.667667	0.707515	0.684007	0.738462	0.621014	0.625663	0.616160	0.650084
13	0.081300	No log	0.713712	0.671713	0.677495	0.670933	0.773913	0.643652	0.633690	0.638331	0.654632
15	0.071900	No log	0.726421	0.692458	0.686580	0.686538	0.761204	0.644520	0.610362	0.614674	0.650606
16	0.068900	No log	0.723746	0.692019	0.686434	0.687053	0.757191	0.638098	0.621231	0.619951	0.653502
18	0.072100	No log	0.723746	0.695094	0.689277	0.691090	0.761204	0.639744	0.618254	0.617961	0.654525
✅ SECTION 10: Model Training Execution completed in 1.3h 19m
🕒 Total runtime so far: 4.3h 18m
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	xlm_roberta	xlm-roberta-base	0.749164	0.710954	0.703824	0.705048	0.760535	0.662129	0.662377	0.66222	0.683634	5.508	271.423	17.066

=== Detailed breakdowns for xlm_roberta ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.802950	0.860045	0.830518	886
1	neutral	0.618902	0.506234	0.556927	401
2	positive	0.711009	0.745192	0.727700	208

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.641892	0.655172	0.648464	435
1	objective	0.505618	0.500000	0.502793	90
2	partisan	0.838877	0.831959	0.835404	970

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.814898	0.605167	0.595376	0.333333	0.886792
1	neutral	401	0.648379	0.629177	0.683544	0.576923	0.627063
2	positive	208	0.745192	0.668153	0.681159	0.518519	0.804781

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	970	0.794845	0.676234	0.876056	0.407547	0.745098
1	non_polarized	435	0.666667	0.682052	0.677871	0.627027	0.741259
2	objective	90	0.655556	0.619100	0.655172	0.702128	0.500000

Saved detailed breakdowns to: ./runs_xlm_roberta_optimized/details
🎯 MULTICLASS CALIBRATION - Optimize prediction biases for better performance
======================================================================

🔧 Calibrating xlm_roberta (xlm-roberta-base)...
📊 Step 1: Extracting polarization logits from trained model...
   Loading model from: ./runs_xlm_roberta_optimized/xlm_roberta
   Warning: No trained weights found at ./runs_xlm_roberta_optimized/xlm_roberta/pytorch_model.bin, using untrained model
   Loading model from: ./runs_xlm_roberta_optimized/xlm_roberta
   Warning: No trained weights found at ./runs_xlm_roberta_optimized/xlm_roberta/pytorch_model.bin, using untrained model
   ✓ Validation logits shape: (1495, 3)
   ✓ Test logits shape: (1495, 3)
🔍 Step 2: Searching for optimal bias vector (coordinate search)...
   ✓ Optimal bias vector found (VAL macro-F1=0.312):
      • non_polarized: +0.50
      •     objective: -0.00
      •      partisan: +0.10
📈 Step 3: Evaluating calibration impact on test set...

   📊 TEST MACRO-F1: 0.259 → 0.305 (+0.046)

   Per-class breakdown:
   📈 non_polarized: P=0.000 R=0.000 F1=0.000 (n=435)  →  P=0.279 R=0.117 F1=0.165 (+0.165)
   📉     objective: P=0.034 R=0.111 F1=0.052 (n=90)  →  P=0.029 R=0.011 F1=0.016 (-0.036)
   📈      partisan: P=0.656 R=0.811 F1=0.725 (n=970)  →  P=0.646 R=0.851 F1=0.734 (+0.009)

✅ Calibration complete! Bias vector saved to:
   ./runs_xlm_roberta_optimized/calibration_vector/xlm_roberta_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!

[xlm_roberta] xlm-roberta-base
Token indices sequence length is longer than the specified maximum sequence length for this model (950 > 512). Running this sequence through the model will result in indexing errors
Token length stats: {'mean': 106.9514, 'p50': 96.0, 'p90': 170.0, 'p95': 182.0, 'p99': 215.0, 'max': 950}
✅ SECTION 11+: Evaluation & Calibration completed in 19.4s
🕒 Total runtime so far: 4.3h 18m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 16.3s
SECTION 3: Configuration Setup           : 1.5h 29m
SECTION 4: Data Loading & Preprocessing  : 8.2s
SECTION 5-9: Model Architecture & Training Setup : 13.1s
SECTION 10: Model Training Execution     : 1.3h 19m
SECTION 11+: Evaluation & Calibration    : 19.4s
======================================== : ==========
TOTAL EXECUTION TIME                     : 4.3h 18m
============================================================