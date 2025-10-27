🚀 Starting SECTION 10: Model Training Execution...

=== Running xlm_roberta -> xlm-roberta-base ===
🔥 Enhanced Oversampling: min=0.90, max=5.84
   ├─ Objective boosted samples: 168 (target: stabilize objective ≈52% F1)
   └─ Neutral boosted samples: 0 (target: keep neutral ≥74% F1)
 [2736/2736 51:56, Epoch 17/18]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	1.297700	No log	0.369576	0.453371	0.506409	0.339543	0.163349	0.523906	0.379723	0.153992	0.246767
1	1.049600	No log	0.553854	0.540708	0.616361	0.534916	0.267483	0.475813	0.441815	0.273503	0.404210
3	0.804100	No log	0.620725	0.644918	0.667323	0.612434	0.417050	0.494024	0.537429	0.411273	0.511854
4	0.643700	No log	0.672792	0.662614	0.704403	0.661073	0.482899	0.516043	0.543082	0.434415	0.547744
6	0.435600	No log	0.689638	0.657592	0.727073	0.677776	0.614599	0.563982	0.626779	0.569686	0.623731
7	0.381200	No log	0.718224	0.707893	0.735905	0.720188	0.620214	0.580034	0.640122	0.588627	0.654407
9	0.288100	No log	0.711588	0.744773	0.717153	0.720088	0.592649	0.594363	0.603728	0.569923	0.645005
10	0.253600	No log	0.734048	0.766019	0.723399	0.740439	0.643696	0.579695	0.630656	0.592735	0.666587
12	0.212000	No log	0.738132	0.761401	0.730876	0.744633	0.656968	0.600374	0.625525	0.610078	0.677356
13	0.206200	No log	0.732006	0.723910	0.743786	0.732798	0.643185	0.598052	0.632799	0.605406	0.669102
15	0.180200	No log	0.737111	0.746046	0.738971	0.742159	0.660541	0.607215	0.631755	0.615875	0.679017
16	0.175700	No log	0.739153	0.750093	0.740293	0.745030	0.663604	0.612005	0.635448	0.620262	0.682646
17	0.167200	No log	0.739153	0.751141	0.740241	0.745454	0.664114	0.612656	0.636207	0.620845	0.683150
✅ SECTION 10: Model Training Execution completed in 52.2m 11s
🕒 Total runtime so far: 59.7m 42s
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	xlm_roberta	xlm-roberta-base	0.733673	0.734482	0.718161	0.725789	0.661224	0.605319	0.637658	0.61571	0.67075	4.1454	472.815	23.641

=== Detailed breakdowns for xlm_roberta ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.734358	0.755079	0.744574	886
1	neutral	0.732247	0.726328	0.729275	866
2	positive	0.736842	0.673077	0.703518	208

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.543590	0.660436	0.596343	642
1	objective	0.465812	0.576720	0.515366	189
2	partisan	0.806554	0.675819	0.735422	1129

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.779910	0.637732	0.568579	0.484848	0.859770
1	neutral	866	0.521940	0.504794	0.599769	0.519174	0.395437
2	positive	208	0.735577	0.669395	0.649351	0.555556	0.803279

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	1129	0.765279	0.736072	0.823872	0.653251	0.731092
1	non_polarized	642	0.696262	0.662949	0.508772	0.774194	0.705882
2	objective	189	0.671958	0.529428	0.395062	0.776557	0.416667

Saved detailed breakdowns to: ./runs_xlm_roberta_run14/details


🎯 MULTICLASS CALIBRATION - Optimize prediction biases for better performance
======================================================================

🔧 Calibrating xlm_roberta (xlm-roberta-base)...
📊 Step 1: Extracting polarization logits from trained model...
   Loading model from: ./runs_xlm_roberta_run14/xlm_roberta
   Loading model from: ./runs_xlm_roberta_run14/xlm_roberta
   ✓ Validation logits shape: (1959, 3)
   ✓ Test logits shape: (1960, 3)
🔍 Step 2: Searching for optimal bias vector (coordinate search)...
   ✓ Optimal bias vector found (VAL macro-F1=0.624):
      • non_polarized: +0.00
      •     objective: -0.70
      •      partisan: +0.10
📈 Step 3: Evaluating calibration impact on test set...

   📊 TEST MACRO-F1: 0.616 → 0.613 (-0.004)

   Per-class breakdown:
   📉 non_polarized: P=0.544 R=0.662 F1=0.597 (n=642)  →  P=0.560 R=0.614 F1=0.585 (-0.012)
   📉     objective: P=0.466 R=0.577 F1=0.515 (n=189)  →  P=0.495 R=0.497 F1=0.496 (-0.019)
   📈      partisan: P=0.807 R=0.676 F1=0.736 (n=1129)  →  P=0.779 R=0.735 F1=0.756 (+0.020)

✅ Calibration complete! Bias vector saved to:
   ./runs_xlm_roberta_run14/calibration_vector/xlm_roberta_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!

Token indices sequence length is longer than the specified maximum sequence length for this model (950 > 512). Running this sequence through the model will result in indexing errors

[xlm_roberta] xlm-roberta-base
Token length stats: {'mean': 107.0766, 'p50': 96.0, 'p90': 170.0, 'p95': 183.0, 'p99': 223.02000000000044, 'max': 950}
✅ SECTION 11+: Evaluation & Calibration completed in 20.4s
🕒 Total runtime so far: 1.0h 0m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 5.2s
SECTION 3: Configuration Setup           : 0.0s
SECTION 4: Data Loading & Preprocessing  : 0.1s
SECTION 5-9: Model Architecture & Training Setup : 5.1m 7s
SECTION 10: Model Training Execution     : 52.2m 11s
SECTION 11+: Evaluation & Calibration    : 20.4s
======================================== : ==========
TOTAL EXECUTION TIME                     : 1.0h 0m
============================================================