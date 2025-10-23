Starting SECTION 10: Model Training Execution...

=== Running xlm_roberta -> xlm-roberta-base ===
🔥 Enhanced Oversampling: min=1.00, max=24.78
   ├─ Objective boosted samples: 405 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 1874 (target: weak class at 49% F1)
 [2900/2900 1:26:55, Epoch 19/20]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	1.166300	No log	0.292308	0.281078	0.401411	0.255615	0.062207	0.020736	0.333333	0.039043	0.147329
1	0.886800	No log	0.416054	0.620879	0.538059	0.437710	0.232107	0.477985	0.455655	0.244064	0.340887
3	0.464500	No log	0.515719	0.620582	0.611130	0.544222	0.606689	0.531207	0.656000	0.532280	0.538251
4	0.300000	No log	0.517057	0.609693	0.629520	0.551695	0.695652	0.568783	0.659493	0.579794	0.565744
6	0.224600	No log	0.591304	0.663562	0.678607	0.626061	0.550502	0.568698	0.600802	0.519445	0.572753
7	0.175500	No log	0.642809	0.695335	0.686300	0.665581	0.654181	0.572289	0.654735	0.582059	0.623820
9	0.129800	No log	0.689632	0.651850	0.690912	0.668190	0.745819	0.611745	0.606656	0.606004	0.637097
10	0.131700	No log	0.605351	0.654756	0.680822	0.632969	0.649498	0.592822	0.587057	0.554823	0.593896
12	0.109500	No log	0.642140	0.669574	0.680239	0.655918	0.707023	0.615812	0.595429	0.583765	0.619842
13	0.085900	No log	0.609365	0.669498	0.681111	0.638560	0.715719	0.626149	0.588745	0.585882	0.612221
15	0.075600	No log	0.657525	0.675024	0.688065	0.667585	0.732441	0.624129	0.609511	0.604394	0.635989
16	0.089900	No log	0.637458	0.691504	0.691523	0.661730	0.723077	0.623631	0.617057	0.605543	0.633636
18	0.065900	No log	0.657525	0.683372	0.694281	0.671792	0.715050	0.631334	0.605905	0.596653	0.634223
19	0.091400	No log	0.656856	0.680471	0.692093	0.670291	0.729766	0.635098	0.612149	0.605196	0.637743
✅ SECTION 10: Model Training Execution completed in 1.5h 27m
🕒 Total runtime so far: 3.3h 21m
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	xlm_roberta	xlm-roberta-base	0.705686	0.694174	0.727125	0.702038	0.745151	0.66685	0.63949	0.641978	0.672008	5.4836	272.63	17.142

=== Detailed breakdowns for xlm_roberta ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.852941	0.687359	0.761250	886
1	neutral	0.509058	0.700748	0.589717	401
2	positive	0.720524	0.793269	0.755149	208

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.578045	0.774713	0.662083	435
1	objective	0.548387	0.377778	0.447368	90
2	partisan	0.874118	0.765979	0.816484	970

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.777652	0.591655	0.569343	0.347826	0.857795
1	neutral	401	0.688279	0.636674	0.745536	0.517647	0.646840
2	positive	208	0.716346	0.609093	0.666667	0.380952	0.779661

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	970	0.716495	0.656509	0.802828	0.403061	0.763636
1	non_polarized	435	0.689655	0.698718	0.589552	0.718280	0.788321
2	objective	90	0.666667	0.617745	0.644068	0.729167	0.480000

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
🕒 Total runtime so far: 3.3h 21m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 10.7s
SECTION 3: Configuration Setup           : 17.6m 35s
SECTION 4: Data Loading & Preprocessing  : 3.6s
SECTION 5-9: Model Architecture & Training Setup : 8.6s
SECTION 10: Model Training Execution     : 1.5h 27m
SECTION 11+: Evaluation & Calibration    : 19.2s
======================================== : ==========
TOTAL EXECUTION TIME                     : 3.3h 21m
============================================================