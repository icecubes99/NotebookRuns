🚀 Starting SECTION 10: Model Training Execution...

=== Running mbert -> bert-base-multilingual-cased ===
🔥 Enhanced Oversampling: min=1.00, max=68.28
   ├─ Objective boosted samples: 405 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 1874 (target: weak class at 49% F1)
 [2900/2900 1:32:23, Epoch 19/20]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	0.978500	No log	0.286288	0.295020	0.384575	0.235865	0.063545	0.087542	0.335025	0.042591	0.139228
1	0.686600	No log	0.408027	0.568599	0.535749	0.429385	0.130435	0.504489	0.380462	0.128457	0.278921
3	0.348100	No log	0.436789	0.609988	0.556253	0.469144	0.408027	0.492556	0.585776	0.389987	0.429566
4	0.201300	No log	0.440134	0.639412	0.573974	0.484345	0.525084	0.517824	0.618648	0.491264	0.487804
6	0.162400	No log	0.456856	0.652119	0.588316	0.506130	0.690301	0.572208	0.625466	0.587083	0.546606
7	0.147900	No log	0.521070	0.636830	0.637712	0.559552	0.520401	0.539794	0.543464	0.469669	0.514610
9	0.147900	No log	0.497659	0.652758	0.619451	0.540052	0.634783	0.566497	0.619309	0.562384	0.551218
10	0.130000	No log	0.529766	0.593383	0.642339	0.553542	0.686288	0.583226	0.600030	0.580209	0.566876
12	0.089000	No log	0.589967	0.655455	0.645144	0.614435	0.723077	0.588657	0.604466	0.594788	0.604612
13	0.077700	No log	0.555184	0.651734	0.647886	0.592412	0.639465	0.578209	0.583451	0.549948	0.571180
15	0.076600	No log	0.579933	0.630295	0.649655	0.605247	0.655518	0.575054	0.585042	0.554509	0.579878
16	0.071100	No log	0.529097	0.645829	0.629761	0.569192	0.570569	0.580890	0.542429	0.491855	0.530523
18	0.083900	No log	0.600000	0.656167	0.656833	0.624036	0.661538	0.576098	0.573458	0.549717	0.586877
19	0.072700	No log	0.581940	0.650051	0.652189	0.611749	0.664883	0.577950	0.573565	0.551861	0.581805
✅ SECTION 10: Model Training Execution completed in 1.6h 33m
🕒 Total runtime so far: 4.3h 19m
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	mbert	bert-base-multilingual-cased	0.61204	0.661771	0.657763	0.629415	0.733779	0.620136	0.633543	0.625407	0.627411	4.4729	334.234	21.015

=== Detailed breakdowns for mbert ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.837838	0.524831	0.645385	886
1	neutral	0.413102	0.770574	0.537859	401
2	positive	0.734375	0.677885	0.705000	208

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.592814	0.682759	0.634615	435
1	objective	0.423913	0.433333	0.428571	90
2	partisan	0.843681	0.784536	0.813034	970

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.788939	0.590720	0.576087	0.327869	0.868206
1	neutral	401	0.628429	0.597698	0.671569	0.515464	0.606061
2	positive	208	0.701923	0.588985	0.675000	0.333333	0.758621

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	970	0.612371	0.590992	0.705781	0.363636	0.703557
1	non_polarized	435	0.600000	0.588119	0.353982	0.674374	0.736000
2	objective	90	0.666667	0.620660	0.571429	0.745098	0.545455

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
      • non_polarized: -0.20
      •     objective: -0.30
      •      partisan: +0.00
📈 Step 3: Evaluating calibration impact on test set...

   📊 TEST MACRO-F1: 0.625 → 0.622 (-0.004)

   Per-class breakdown:
   📉 non_polarized: P=0.593 R=0.683 F1=0.635 (n=435)  →  P=0.688 R=0.508 F1=0.585 (-0.050)
   📈     objective: P=0.424 R=0.433 F1=0.429 (n=90)  →  P=0.448 R=0.433 F1=0.441 (+0.012)
   📈      partisan: P=0.844 R=0.785 F1=0.813 (n=970)  →  P=0.795 R=0.891 F1=0.840 (+0.027)

✅ Calibration complete! Bias vector saved to:
   ./runs_mbert_optimized/calibration_vector/mbert_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!

[mbert] bert-base-multilingual-cased
Token indices sequence length is longer than the specified maximum sequence length for this model (916 > 512). Running this sequence through the model will result in indexing errors
Token length stats: {'mean': 109.174, 'p50': 97.0, 'p90': 179.0, 'p95': 194.0, 'p99': 226.02000000000044, 'max': 916}
✅ SECTION 11+: Evaluation & Calibration completed in 14.5s
🕒 Total runtime so far: 4.3h 20m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 9.2s
SECTION 3: Configuration Setup           : 15.8m 49s
SECTION 4: Data Loading & Preprocessing  : 1.2m 11s
SECTION 5-9: Model Architecture & Training Setup : 53.2s
SECTION 10: Model Training Execution     : 1.6h 33m
SECTION 11+: Evaluation & Calibration    : 14.5s
======================================== : ==========
TOTAL EXECUTION TIME                     : 4.3h 20m
============================================================