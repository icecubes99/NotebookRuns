🚀 Starting SECTION 10: Model Training Execution...

=== Running mbert -> bert-base-multilingual-cased ===
🔥 Enhanced Oversampling: min=1.00, max=68.28
   ├─ Objective boosted samples: 405 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 1874 (target: weak class at 49% F1)
 [2900/2900 1:09:56, Epoch 19/20]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	0.969300	No log	0.290301	0.292898	0.395799	0.250409	0.062207	0.020736	0.333333	0.039043	0.144726
1	0.710200	No log	0.411371	0.629624	0.527647	0.438483	0.127090	0.514514	0.376232	0.121295	0.279889
3	0.319100	No log	0.438127	0.594566	0.546352	0.458398	0.333779	0.490500	0.558507	0.321603	0.390001
4	0.214500	No log	0.527759	0.658857	0.566103	0.548146	0.507692	0.515384	0.621873	0.481161	0.514653
6	0.162200	No log	0.446154	0.658269	0.576172	0.488032	0.648829	0.556537	0.624000	0.564950	0.526491
7	0.136100	No log	0.529097	0.617152	0.629183	0.558428	0.592642	0.551954	0.578278	0.524904	0.541666
9	0.112800	No log	0.503010	0.659670	0.607120	0.540867	0.634114	0.558533	0.574814	0.536330	0.538598
10	0.110800	No log	0.493645	0.583895	0.633879	0.511696	0.680936	0.577974	0.579248	0.563235	0.537465
12	0.079000	No log	0.609365	0.632713	0.658883	0.622734	0.703010	0.568739	0.605367	0.580283	0.601509
13	0.073600	No log	0.577926	0.634636	0.660507	0.604148	0.650836	0.576140	0.579312	0.549236	0.576692
15	0.073400	No log	0.569231	0.623668	0.661583	0.592306	0.622742	0.564593	0.564231	0.528330	0.560318
16	0.071500	No log	0.575251	0.644953	0.660735	0.603045	0.595987	0.571767	0.554288	0.510727	0.556886
18	0.086800	No log	0.593311	0.652943	0.663771	0.620535	0.692977	0.600369	0.583170	0.569474	0.595004
19	0.069800	No log	0.584615	0.653479	0.663160	0.616182	0.688294	0.594060	0.580855	0.565373	0.590778
✅ SECTION 10: Model Training Execution completed in 1.2h 10m
🕒 Total runtime so far: 9.3h 20m
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	mbert	bert-base-multilingual-cased	0.610702	0.637009	0.661538	0.625369	0.703679	0.592552	0.628624	0.60429	0.614829	4.5974	325.186	20.446

=== Detailed breakdowns for mbert ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.813149	0.530474	0.642077	886
1	neutral	0.420749	0.728180	0.533333	401
2	positive	0.677130	0.725962	0.700696	208

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.556738	0.721839	0.628629	435
1	objective	0.373832	0.444444	0.406091	90
2	partisan	0.847087	0.719588	0.778149	970

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.749436	0.546646	0.552764	0.250000	0.837174
1	neutral	401	0.635910	0.599406	0.697674	0.520833	0.579710
2	positive	208	0.639423	0.573766	0.631579	0.413793	0.675926

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	970	0.611340	0.579641	0.703578	0.346939	0.688406
1	non_polarized	435	0.604598	0.598440	0.360515	0.677228	0.757576
2	objective	90	0.633333	0.589352	0.526316	0.720000	0.521739

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
   ✓ Optimal bias vector found (VAL macro-F1=0.599):
      • non_polarized: -0.20
      •     objective: +0.70
      •      partisan: +0.00
📈 Step 3: Evaluating calibration impact on test set...

   📊 TEST MACRO-F1: 0.604 → 0.615 (+0.011)

   Per-class breakdown:
   📉 non_polarized: P=0.557 R=0.722 F1=0.629 (n=435)  →  P=0.667 R=0.584 F1=0.623 (-0.006)
   📉     objective: P=0.374 R=0.444 F1=0.406 (n=90)  →  P=0.353 R=0.467 F1=0.402 (-0.004)
   📈      partisan: P=0.847 R=0.720 F1=0.778 (n=970)  →  P=0.810 R=0.831 F1=0.820 (+0.042)

✅ Calibration complete! Bias vector saved to:
   ./runs_mbert_optimized/calibration_vector/mbert_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!

[mbert] bert-base-multilingual-cased
Token indices sequence length is longer than the specified maximum sequence length for this model (916 > 512). Running this sequence through the model will result in indexing errors
Token length stats: {'mean': 109.174, 'p50': 97.0, 'p90': 179.0, 'p95': 194.0, 'p99': 226.02000000000044, 'max': 916}
✅ SECTION 11+: Evaluation & Calibration completed in 44.8s
🕒 Total runtime so far: 9.3h 20m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 9.2s
SECTION 3: Configuration Setup           : 5.3m 18s
SECTION 4: Data Loading & Preprocessing  : 1.2m 11s
SECTION 5-9: Model Architecture & Training Setup : 5.3s
SECTION 10: Model Training Execution     : 1.2h 10m
SECTION 11+: Evaluation & Calibration    : 44.8s
======================================== : ==========
TOTAL EXECUTION TIME                     : 9.3h 20m
============================================================