🚀 Starting SECTION 10: Model Training Execution...

=== Running xlm_roberta -> xlm-roberta-base ===
🔥 Enhanced Oversampling: min=1.00, max=26.43
   ├─ Objective boosted samples: 405 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 227 (target: weak class at 49% F1)
 [2906/3190 1:26:16 < 08:26, 0.56 it/s, Epoch 19/22]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	1.157200	No log	0.290301	0.259073	0.404391	0.259632	0.062207	0.020736	0.333333	0.039043	0.149337
1	0.816000	No log	0.412040	0.515781	0.561833	0.414876	0.212040	0.480042	0.453818	0.231807	0.323341
3	0.429800	No log	0.485619	0.596203	0.614024	0.511918	0.568562	0.517630	0.645527	0.503359	0.507638
4	0.303600	No log	0.619398	0.626625	0.658513	0.624039	0.565886	0.518244	0.630544	0.509696	0.566868
6	0.235100	No log	0.471572	0.640452	0.618353	0.519806	0.638127	0.567176	0.628308	0.567813	0.543810
7	0.172700	No log	0.610702	0.649704	0.667165	0.631657	0.613378	0.578417	0.600190	0.550404	0.591031
9	0.117000	No log	0.595318	0.693416	0.664539	0.630498	0.664883	0.603167	0.600301	0.573405	0.601952
10	0.099000	No log	0.583278	0.669228	0.670639	0.621206	0.678930	0.572024	0.606346	0.573718	0.597462
12	0.093300	No log	0.644147	0.683024	0.675322	0.659626	0.678261	0.596096	0.594739	0.573537	0.616582
13	0.082600	No log	0.662207	0.675459	0.677655	0.665356	0.774582	0.662938	0.604577	0.625703	0.645529
15	0.063200	No log	0.636789	0.656212	0.685868	0.649158	0.748495	0.630241	0.597734	0.601761	0.625459
16	0.058500	No log	0.645485	0.674366	0.674466	0.657977	0.743813	0.638436	0.607595	0.608712	0.633345
18	0.057400	No log	0.663545	0.674361	0.686140	0.669421	0.701672	0.627137	0.603414	0.589901	0.629661
19	0.074900	No log	0.654849	0.674005	0.684873	0.665283	0.718395	0.640179	0.604143	0.595545	0.630414
✅ SECTION 10: Model Training Execution completed in 1.4h 27m
🕒 Total runtime so far: 5.1h 3m
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	xlm_roberta	xlm-roberta-base	0.708361	0.711787	0.723964	0.708229	0.759197	0.655134	0.613783	0.630331	0.66928	5.6829	263.07	16.541

=== Detailed breakdowns for xlm_roberta ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.847978	0.686230	0.758578	886
1	neutral	0.511265	0.735661	0.603272	401
2	positive	0.776119	0.750000	0.762836	208

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.636574	0.632184	0.634371	435
1	objective	0.500000	0.355556	0.415584	90
2	partisan	0.828829	0.853608	0.841036	970

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.827314	0.588279	0.574194	0.292683	0.897959
1	neutral	401	0.645885	0.611149	0.676768	0.511111	0.645570
2	positive	208	0.687500	0.555151	0.645963	0.260870	0.758621

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	970	0.716495	0.657420	0.803432	0.413965	0.754864
1	non_polarized	435	0.689655	0.701710	0.562738	0.724211	0.818182
2	objective	90	0.711111	0.646495	0.655172	0.784314	0.500000

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
🕒 Total runtime so far: 5.1h 4m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 10.7s
SECTION 3: Configuration Setup           : 15.8m 49s
SECTION 4: Data Loading & Preprocessing  : 3.6s
SECTION 5-9: Model Architecture & Training Setup : 8.6s
SECTION 10: Model Training Execution     : 1.4h 27m
SECTION 11+: Evaluation & Calibration    : 19.2s
======================================== : ==========
TOTAL EXECUTION TIME                     : 5.1h 4m
============================================================