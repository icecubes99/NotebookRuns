🚀 Starting SECTION 10: Model Training Execution...

=== Running mbert -> bert-base-multilingual-cased ===
🔥 Enhanced Oversampling: min=1.00, max=68.85
   ├─ Objective boosted samples: 405 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 1874 (target: weak class at 49% F1)
 [2325/2900 1:10:08 < 17:21, 0.55 it/s, Epoch 15/20]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	1.044200	No log	0.278261	0.651186	0.356206	0.186718	0.062207	0.020736	0.333333	0.039043	0.112880
1	0.693700	No log	0.363211	0.623107	0.449226	0.351764	0.190635	0.452656	0.449936	0.205962	0.278863
3	0.363800	No log	0.380602	0.566037	0.568532	0.386533	0.654181	0.569663	0.617447	0.512929	0.449731
4	0.226100	No log	0.451505	0.598728	0.584354	0.476762	0.670234	0.552181	0.652627	0.569738	0.523250
6	0.163800	No log	0.608696	0.635202	0.626735	0.616036	0.609365	0.544307	0.616988	0.546056	0.581046
7	0.142700	No log	0.527759	0.673480	0.589892	0.555927	0.751171	0.622926	0.618616	0.616609	0.586268
9	0.131800	No log	0.450836	0.570964	0.607864	0.467121	0.536455	0.548469	0.578428	0.502245	0.484683
10	0.100900	No log	0.494314	0.644747	0.599918	0.535056	0.591973	0.558817	0.592861	0.532477	0.533767
12	0.080500	No log	0.534448	0.657590	0.622695	0.572781	0.647492	0.590081	0.587581	0.560832	0.566807
13	0.076600	No log	0.537793	0.669803	0.614828	0.573213	0.553177	0.592417	0.554191	0.507012	0.540113
15	0.077100	No log	0.515719	0.642079	0.627547	0.555948	0.583946	0.588541	0.568533	0.526042	0.540995
✅ SECTION 10: Model Training Execution completed in 1.2h 10m
🕒 Total runtime so far: 2.8h 48m
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	mbert	bert-base-multilingual-cased	0.545819	0.679696	0.602244	0.571437	0.722408	0.602572	0.60733	0.599457	0.585447	4.6048	324.659	20.413

=== Detailed breakdowns for mbert ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.852029	0.402935	0.547126	886
1	neutral	0.369542	0.865337	0.517910	401
2	positive	0.817518	0.538462	0.649275	208

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.645833	0.498851	0.562905	435
1	objective	0.377193	0.477778	0.421569	90
2	partisan	0.784689	0.845361	0.813896	970

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.810384	0.584491	0.573477	0.292683	0.887314
1	neutral	401	0.551122	0.548129	0.532164	0.540000	0.572222
2	positive	208	0.677885	0.574290	0.613333	0.363636	0.745902

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	970	0.508247	0.512161	0.602844	0.329866	0.603774
1	non_polarized	435	0.604598	0.568242	0.246154	0.697509	0.761062
2	objective	90	0.666667	0.602453	0.545455	0.761905	0.500000

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
   ✓ Optimal bias vector found (VAL macro-F1=0.329):
      • non_polarized: -0.50
      •     objective: -0.80
      •      partisan: +0.00
📈 Step 3: Evaluating calibration impact on test set...

   📊 TEST MACRO-F1: 0.292 → 0.262 (-0.029)

   Per-class breakdown:
   📉 non_polarized: P=0.273 R=0.007 F1=0.013 (n=435)  →  P=0.000 R=0.000 F1=0.000 (-0.013)
   📉     objective: P=0.095 R=0.133 F1=0.111 (n=90)  →  P=0.000 R=0.000 F1=0.000 (-0.111)
   📈      partisan: P=0.644 R=0.901 F1=0.751 (n=970)  →  P=0.649 R=1.000 F1=0.787 (+0.036)

✅ Calibration complete! Bias vector saved to:
   ./runs_mbert_optimized/calibration_vector/mbert_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!

[mbert] bert-base-multilingual-cased
Token indices sequence length is longer than the specified maximum sequence length for this model (916 > 512). Running this sequence through the model will result in indexing errors
Token length stats: {'mean': 109.174, 'p50': 97.0, 'p90': 179.0, 'p95': 194.0, 'p99': 226.02000000000044, 'max': 916}
✅ SECTION 11+: Evaluation & Calibration completed in 14.2s
🕒 Total runtime so far: 2.8h 48m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 18.3s
SECTION 3: Configuration Setup           : 30.3m 20s
SECTION 4: Data Loading & Preprocessing  : 3.4m 21s
SECTION 5-9: Model Architecture & Training Setup : 10.3s
SECTION 10: Model Training Execution     : 1.2h 10m
SECTION 11+: Evaluation & Calibration    : 14.2s
======================================== : ==========
TOTAL EXECUTION TIME                     : 2.8h 48m
============================================================