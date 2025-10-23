## SECTION 10

🚀 Starting SECTION 10: Model Training Execution...

=== Running xlm_roberta -> xlm-roberta-base ===
🔥 Enhanced Oversampling: min=1.00, max=24.78
   ├─ Objective boosted samples: 405 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 1874 (target: weak class at 49% F1)
 [2900/2900 1:22:58, Epoch 19/20]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	1.190900	No log	0.288963	0.280674	0.391016	0.242266	0.062207	0.020736	0.333333	0.039043	0.140655
1	0.914500	No log	0.394649	0.655011	0.513726	0.413081	0.234783	0.484193	0.460585	0.250045	0.331563
3	0.468700	No log	0.494314	0.651809	0.606463	0.534212	0.612040	0.529466	0.647501	0.530032	0.532122
4	0.345700	No log	0.559197	0.634894	0.648942	0.592478	0.713043	0.573707	0.634010	0.587720	0.590099
6	0.225100	No log	0.550502	0.637098	0.660016	0.589834	0.527759	0.556973	0.580990	0.497299	0.543567
7	0.168800	No log	0.536455	0.670256	0.638101	0.581018	0.662207	0.580670	0.634383	0.586328	0.583673
9	0.132900	No log	0.680936	0.642358	0.692566	0.661535	0.737124	0.631410	0.598811	0.598229	0.629882
10	0.135300	No log	0.585953	0.664534	0.673004	0.622646	0.703679	0.623744	0.600798	0.589035	0.605840
12	0.145300	No log	0.614047	0.670204	0.669342	0.638599	0.703010	0.603568	0.596699	0.581261	0.609930
13	0.083000	No log	0.564548	0.665419	0.661136	0.605121	0.710368	0.636470	0.595021	0.589433	0.597277
15	0.076900	No log	0.646823	0.668534	0.685216	0.659123	0.731773	0.640779	0.626317	0.619610	0.639367
16	0.086300	No log	0.639465	0.687770	0.691620	0.661916	0.725084	0.634834	0.619756	0.612117	0.637017
18	0.066800	No log	0.643478	0.676425	0.687287	0.661416	0.720401	0.634084	0.609581	0.600694	0.631055
19	0.087400	No log	0.647492	0.676612	0.689091	0.663631	0.729097	0.637230	0.611818	0.605595	0.634613
✅ SECTION 10: Model Training Execution completed in 1.4h 23m
🕒 Total runtime so far: 4.2h 13m
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	xlm_roberta	xlm-roberta-base	0.679599	0.687707	0.717874	0.686096	0.740468	0.656583	0.642959	0.640642	0.663369	5.7836	258.49	16.253


## SECTION 11+

=== Detailed breakdowns for xlm_roberta ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.861371	0.624153	0.723822	886
1	neutral	0.478537	0.750623	0.584466	401
2	positive	0.723214	0.778846	0.750000	208

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.569728	0.770115	0.654936	435
1	objective	0.521739	0.400000	0.452830	90
2	partisan	0.878282	0.758763	0.814159	970

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.776524	0.605650	0.569378	0.391304	0.856269
1	neutral	401	0.673317	0.628896	0.732143	0.533333	0.621212
2	positive	208	0.716346	0.573300	0.662420	0.260870	0.796610

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	970	0.683505	0.642095	0.769731	0.398190	0.758364
1	non_polarized	435	0.664368	0.664789	0.512195	0.710744	0.771429
2	objective	90	0.711111	0.650570	0.641509	0.788462	0.521739

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
   ✓ Optimal bias vector found (VAL macro-F1=0.281):
      • non_polarized: +0.00
      •     objective: +0.40
      •      partisan: +0.00
📈 Step 3: Evaluating calibration impact on test set...

   📊 TEST MACRO-F1: 0.150 → 0.173 (+0.023)

   Per-class breakdown:
   📉 non_polarized: P=0.291 R=1.000 F1=0.451 (n=435)  →  P=0.292 R=0.986 F1=0.451 (-0.000)
   📈     objective: P=0.000 R=0.000 F1=0.000 (n=90)  →  P=0.154 R=0.044 F1=0.069 (+0.069)
   ➡️      partisan: P=0.000 R=0.000 F1=0.000 (n=970)  →  P=0.000 R=0.000 F1=0.000 (+0.000)

✅ Calibration complete! Bias vector saved to:
   ./runs_xlm_roberta_optimized/calibration_vector/xlm_roberta_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!


[xlm_roberta] xlm-roberta-base
Token indices sequence length is longer than the specified maximum sequence length for this model (950 > 512). Running this sequence through the model will result in indexing errors
Token length stats: {'mean': 106.9514, 'p50': 96.0, 'p90': 170.0, 'p95': 182.0, 'p99': 215.0, 'max': 950}
✅ SECTION 11+: Evaluation & Calibration completed in 19.2s
🕒 Total runtime so far: 4.2h 13m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 9.9s
SECTION 5-9: Model Architecture & Training Setup : 3.2s
SECTION 3: Configuration Setup           : 15.7m 41s
SECTION 4: Data Loading & Preprocessing  : 3.3s
SECTION 10: Model Training Execution     : 1.4h 23m
SECTION 11+: Evaluation & Calibration    : 19.2s
======================================== : ==========
TOTAL EXECUTION TIME                     : 4.2h 13m
============================================================

