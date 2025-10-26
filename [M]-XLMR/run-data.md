🚀 Starting SECTION 10: Model Training Execution...

=== Running xlm_roberta -> xlm-roberta-base ===
🔥 Enhanced Oversampling: min=0.50, max=17.00
   ├─ Objective boosted samples: 405 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 227 (target: weak class at 49% F1)
 [2470/2900 1:49:32 < 19:05, 0.38 it/s, Epoch 16/20]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	1.229500	No log	0.268227	0.458005	0.421361	0.256676	0.073579	0.162973	0.347716	0.066165	0.161421
1	0.866700	No log	0.486957	0.501050	0.565578	0.467626	0.272241	0.465253	0.478650	0.282268	0.374947
3	0.460100	No log	0.547826	0.556985	0.633598	0.540613	0.468227	0.495441	0.612899	0.436674	0.488644
4	0.332500	No log	0.634114	0.587884	0.661280	0.597699	0.650167	0.548955	0.673534	0.562414	0.580057
6	0.227200	No log	0.704348	0.676656	0.675039	0.675804	0.700334	0.590446	0.656778	0.610999	0.643401
7	0.159800	No log	0.622074	0.636388	0.683433	0.632581	0.635452	0.562021	0.631593	0.566260	0.599420
9	0.139200	No log	0.680268	0.675314	0.694265	0.678490	0.680936	0.576609	0.635235	0.587967	0.633229
10	0.113900	No log	0.688963	0.671982	0.694502	0.679062	0.748495	0.609784	0.626792	0.617258	0.648160
12	0.095300	No log	0.694983	0.671284	0.683189	0.676499	0.706355	0.590593	0.598644	0.585711	0.631105
13	0.102400	No log	0.693645	0.675856	0.703102	0.684773	0.733110	0.620597	0.596473	0.594619	0.639696
15	0.075600	No log	0.691639	0.675274	0.694523	0.681889	0.725753	0.607511	0.600212	0.590909	0.636399
16	0.075900	No log	0.705017	0.683633	0.692332	0.687386	0.748495	0.620205	0.603210	0.603488	0.645437
✅ SECTION 10: Model Training Execution completed in 1.8h 50m
🕒 Total runtime so far: 6.4h 26m
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	xlm_roberta	xlm-roberta-base	0.717057	0.69862	0.725431	0.706685	0.738462	0.629502	0.652727	0.639683	0.673184	5.7617	259.473	16.315

=== Detailed breakdowns for xlm_roberta ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.845950	0.718962	0.777303	886
1	neutral	0.527938	0.683292	0.595652	401
2	positive	0.721973	0.774038	0.747100	208

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.595876	0.664368	0.628261	435
1	objective	0.445545	0.500000	0.471204	90
2	partisan	0.847085	0.793814	0.819585	970

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.787810	0.582201	0.568306	0.310345	0.867953
1	neutral	401	0.645885	0.630277	0.668342	0.584906	0.637584
2	positive	208	0.706731	0.603190	0.666667	0.370370	0.772532

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	970	0.739175	0.664933	0.825324	0.411111	0.758364
1	non_polarized	435	0.666667	0.677652	0.578571	0.691796	0.762590
2	objective	90	0.722222	0.651360	0.625000	0.807339	0.521739

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
   ✓ Optimal bias vector found (VAL macro-F1=0.357):
      • non_polarized: +0.60
      •     objective: +0.00
      •      partisan: +0.80
📈 Step 3: Evaluating calibration impact on test set...

   📊 TEST MACRO-F1: 0.150 → 0.150 (+0.000)

   Per-class breakdown:
   ➡️ non_polarized: P=0.291 R=1.000 F1=0.451 (n=435)  →  P=0.291 R=1.000 F1=0.451 (+0.000)
   ➡️     objective: P=0.000 R=0.000 F1=0.000 (n=90)  →  P=0.000 R=0.000 F1=0.000 (+0.000)
   ➡️      partisan: P=0.000 R=0.000 F1=0.000 (n=970)  →  P=0.000 R=0.000 F1=0.000 (+0.000)

✅ Calibration complete! Bias vector saved to:
   ./runs_xlm_roberta_optimized/calibration_vector/xlm_roberta_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!

[xlm_roberta] xlm-roberta-base
Token indices sequence length is longer than the specified maximum sequence length for this model (950 > 512). Running this sequence through the model will result in indexing errors
Token length stats: {'mean': 106.9514, 'p50': 96.0, 'p90': 170.0, 'p95': 182.0, 'p99': 215.0, 'max': 950}
✅ SECTION 11+: Evaluation & Calibration completed in 19.3s
🕒 Total runtime so far: 6.4h 27m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 16.0s
SECTION 3: Configuration Setup           : 9.8m 51s
SECTION 4: Data Loading & Preprocessing  : 0.1s
SECTION 5-9: Model Architecture & Training Setup : 4.1s
SECTION 10: Model Training Execution     : 1.8h 50m
SECTION 11+: Evaluation & Calibration    : 19.3s
======================================== : ==========
TOTAL EXECUTION TIME                     : 6.4h 27m
============================================================