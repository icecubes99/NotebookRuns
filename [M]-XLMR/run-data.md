🚀 Starting SECTION 10: Model Training Execution...

=== Running xlm_roberta -> xlm-roberta-base ===
tokenizer_config.json:   0%|          | 0.00/25.0 [00:00<?, ?B/s]config.json:   0%|          | 0.00/615 [00:00<?, ?B/s]sentencepiece.bpe.model:   0%|          | 0.00/5.07M [00:00<?, ?B/s]tokenizer.json:   0%|          | 0.00/9.10M [00:00<?, ?B/s]model.safetensors:   0%|          | 0.00/1.12G [00:00<?, ?B/s]
🔥 Enhanced Oversampling: min=0.30, max=14.88
   ├─ Objective boosted samples: 405 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 227 (target: weak class at 49% F1)
 [2900/2900 1:27:48, Epoch 19/20]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	1.166600	No log	0.303679	0.276912	0.402813	0.234027	0.080936	0.142506	0.354284	0.081460	0.157743
1	0.844000	No log	0.594649	0.525833	0.568028	0.502581	0.433445	0.483025	0.532579	0.366600	0.434591
3	0.477400	No log	0.549164	0.577146	0.557443	0.442580	0.591973	0.521817	0.638965	0.520148	0.481364
4	0.338700	No log	0.607358	0.600266	0.672018	0.606683	0.429431	0.509382	0.585243	0.416096	0.511390
6	0.188200	No log	0.653512	0.603903	0.674954	0.622039	0.737124	0.604573	0.629193	0.615533	0.618786
7	0.181800	No log	0.658863	0.628423	0.681852	0.642886	0.659532	0.589845	0.632060	0.583396	0.613141
9	0.131500	No log	0.723077	0.693366	0.672483	0.680021	0.772575	0.637719	0.630998	0.633882	0.656952
10	0.104900	No log	0.724415	0.678992	0.674784	0.665489	0.767893	0.645918	0.596017	0.605419	0.635454
12	0.094500	No log	0.707023	0.661321	0.685963	0.671694	0.741806	0.637636	0.600449	0.599174	0.635434
13	0.071800	No log	0.724415	0.685586	0.673143	0.673855	0.761873	0.636358	0.613562	0.620993	0.647424
15	0.087700	No log	0.739130	0.715450	0.683482	0.693126	0.783946	0.666424	0.616583	0.635051	0.664089
16	0.066900	No log	0.733779	0.698476	0.687345	0.688946	0.763211	0.647456	0.621630	0.624579	0.656762
18	0.087400	No log	0.733110	0.704516	0.687343	0.693422	0.769231	0.651632	0.626829	0.630938	0.662180
19	0.072700	No log	0.731104	0.701377	0.690041	0.693827	0.766555	0.652204	0.626022	0.629876	0.661852
✅ SECTION 10: Model Training Execution completed in 1.5h 29m
🕒 Total runtime so far: 1.5h 29m
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	xlm_roberta	xlm-roberta-base	0.75786	0.739786	0.705213	0.718324	0.76388	0.662324	0.627557	0.641737	0.680031	5.7785	258.72	16.267

=== Detailed breakdowns for xlm_roberta ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.788889	0.881490	0.832623	886
1	neutral	0.636656	0.493766	0.556180	401
2	positive	0.793814	0.740385	0.766169	208

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.644144	0.657471	0.650739	435
1	objective	0.507463	0.377778	0.433121	90
2	partisan	0.835366	0.847423	0.841351	970

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.834086	0.637469	0.623529	0.390244	0.898634
1	neutral	401	0.630923	0.595481	0.670025	0.494624	0.621795
2	positive	208	0.721154	0.573218	0.661972	0.260870	0.796813

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	970	0.810309	0.684264	0.886584	0.388430	0.777778
1	non_polarized	435	0.655172	0.684669	0.650407	0.612903	0.790698
2	objective	90	0.688889	0.629010	0.655738	0.755102	0.476190

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
✅ SECTION 11+: Evaluation & Calibration completed in 24.1s
🕒 Total runtime so far: 1.5h 30m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 16.3s
SECTION 3: Configuration Setup           : 0.0s
SECTION 4: Data Loading & Preprocessing  : 8.2s
SECTION 5-9: Model Architecture & Training Setup : 20.1s
SECTION 10: Model Training Execution     : 1.5h 29m
SECTION 11+: Evaluation & Calibration    : 24.1s
======================================== : ==========
TOTAL EXECUTION TIME                     : 1.5h 30m
============================================================