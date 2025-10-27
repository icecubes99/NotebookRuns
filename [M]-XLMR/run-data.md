🚀 Starting SECTION 10: Model Training Execution...

=== Running xlm_roberta -> xlm-roberta-base ===
🔥 Enhanced Oversampling: min=0.90, max=5.84
   ├─ Objective boosted samples: 168 (target: stabilize objective ≈52% F1)
   └─ Neutral boosted samples: 0 (target: keep neutral ≥74% F1)
 [2736/2736 1:13:14, Epoch 17/18]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	1.287500	No log	0.282797	0.471818	0.448578	0.273914	0.158244	0.448380	0.372645	0.146419	0.210167
1	0.998500	No log	0.560490	0.539825	0.632446	0.542362	0.340480	0.458431	0.478888	0.339971	0.441166
3	0.789900	No log	0.624809	0.616053	0.688192	0.612172	0.402757	0.497498	0.526167	0.392293	0.502233
4	0.630000	No log	0.675345	0.671716	0.697629	0.665676	0.497192	0.518974	0.558663	0.453052	0.559364
6	0.447900	No log	0.698826	0.669016	0.727519	0.690056	0.610516	0.552837	0.609141	0.554530	0.622293
7	0.422300	No log	0.713119	0.720004	0.726146	0.722080	0.609495	0.566530	0.621908	0.574050	0.648065
9	0.267900	No log	0.716182	0.747343	0.722911	0.726394	0.603880	0.599622	0.616634	0.581731	0.654063
10	0.262700	No log	0.732006	0.748022	0.734896	0.741111	0.645738	0.583717	0.625461	0.597777	0.669444
12	0.210400	No log	0.735069	0.752646	0.730705	0.739574	0.660031	0.610083	0.628475	0.617393	0.678484
13	0.230200	No log	0.730985	0.732099	0.742921	0.737253	0.649821	0.607584	0.637626	0.614270	0.675761
15	0.184100	No log	0.737111	0.754645	0.736259	0.744860	0.661562	0.611192	0.627948	0.618000	0.681430
16	0.184800	No log	0.740174	0.755384	0.739612	0.746627	0.660031	0.607711	0.626358	0.615419	0.681023
17	0.167700	No log	0.741705	0.757416	0.739506	0.747427	0.656968	0.605229	0.625699	0.613251	0.680339
✅ SECTION 10: Model Training Execution completed in 1.3h 16m
🕒 Total runtime so far: 2.5h 30m
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	xlm_roberta	xlm-roberta-base	0.733163	0.73836	0.714106	0.725106	0.670408	0.613173	0.640932	0.622791	0.673949	4.1907	467.7	23.385

=== Detailed breakdowns for xlm_roberta ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.730937	0.757336	0.743902	886
1	neutral	0.731395	0.726328	0.728853	866
2	positive	0.752747	0.658654	0.702564	208

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.558747	0.666667	0.607955	642
1	objective	0.477679	0.566138	0.518160	189
2	partisan	0.803093	0.689991	0.742258	1129

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.785553	0.647311	0.571429	0.507937	0.862566
1	neutral	866	0.535797	0.519403	0.612865	0.521212	0.424132
2	positive	208	0.740385	0.659773	0.670807	0.500000	0.808511

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	1129	0.759965	0.730427	0.818909	0.648233	0.724138
1	non_polarized	642	0.704050	0.670020	0.534483	0.778607	0.696970
2	objective	189	0.671958	0.561247	0.370370	0.774908	0.538462

Saved detailed breakdowns to: ./runs_xlm_roberta_run15/details

🎯 MULTICLASS CALIBRATION - Optimize prediction biases for better performance
======================================================================

🔧 Calibrating xlm_roberta (xlm-roberta-base)...
📊 Step 1: Extracting polarization logits from trained model...
   Loading model from: ./runs_xlm_roberta_run15/xlm_roberta
   Loading model from: ./runs_xlm_roberta_run15/xlm_roberta
   ✓ Validation logits shape: (1959, 3)
   ✓ Test logits shape: (1960, 3)
🔍 Step 2: Searching for optimal bias vector (coordinate search)...
   ✓ Optimal bias vector found (VAL macro-F1=0.619):
      • non_polarized: +0.00
      •     objective: -0.40
      •      partisan: +0.00
📈 Step 3: Evaluating calibration impact on test set...

   ⚠️ TEST MACRO-F1: 0.622 → 0.622 (-0.001) < threshold (+0.002); keeping raw logits.

   Per-class breakdown:
   ➡️ non_polarized: P=0.558 R=0.667 F1=0.608 (n=642)  →  P=0.558 R=0.667 F1=0.608 (+0.000)
   ➡️     objective: P=0.478 R=0.566 F1=0.518 (n=189)  →  P=0.478 R=0.566 F1=0.518 (+0.000)
   ➡️      partisan: P=0.803 R=0.689 F1=0.742 (n=1129)  →  P=0.803 R=0.689 F1=0.742 (+0.000)

✅ Calibration complete! Bias vector saved to:
   ./runs_xlm_roberta_run15/calibration_vector/xlm_roberta_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!

[xlm_roberta] xlm-roberta-base
Token length stats: {'mean': 107.0766, 'p50': 96.0, 'p90': 170.0, 'p95': 183.0, 'p99': 223.02000000000044, 'max': 950}
✅ SECTION 11+: Evaluation & Calibration completed in 20.7s
🕒 Total runtime so far: 2.5h 30m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 5.2s
SECTION 3: Configuration Setup           : 1.0m 1s
SECTION 4: Data Loading & Preprocessing  : 0.1s
SECTION 5-9: Model Architecture & Training Setup : 22.2s
SECTION 10: Model Training Execution     : 1.3h 16m
SECTION 11+: Evaluation & Calibration    : 20.7s
======================================== : ==========
TOTAL EXECUTION TIME                     : 2.5h 30m
============================================================