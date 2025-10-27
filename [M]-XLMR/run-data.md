🚀 Starting SECTION 10: Model Training Execution...

=== Running xlm_roberta -> xlm-roberta-base ===
tokenizer_config.json:   0%|          | 0.00/25.0 [00:00<?, ?B/s]config.json:   0%|          | 0.00/615 [00:00<?, ?B/s]sentencepiece.bpe.model:   0%|          | 0.00/5.07M [00:00<?, ?B/s]tokenizer.json:   0%|          | 0.00/9.10M [00:00<?, ?B/s]model.safetensors:   0%|          | 0.00/1.12G [00:00<?, ?B/s]
🔥 Enhanced Oversampling: min=0.80, max=5.57
   ├─ Objective boosted samples: 168 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 0 (target: weak class at 49% F1)
 [1566/2736 39:49 < 29:47, 0.65 it/s, Epoch 10.25/18]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	1.279600	No log	0.367534	0.474515	0.482856	0.339291	0.143951	0.430584	0.353518	0.123606	0.231449
1	1.035900	No log	0.559469	0.550479	0.622073	0.525183	0.374170	0.473673	0.487131	0.365398	0.445291
3	0.772500	No log	0.656457	0.621253	0.704262	0.640881	0.464523	0.499845	0.534101	0.434867	0.537874
4	0.708300	No log	0.650842	0.617073	0.707545	0.632411	0.503318	0.516408	0.565323	0.468859	0.550635
6	0.414600	No log	0.681981	0.668373	0.714469	0.677682	0.594691	0.554462	0.624706	0.557105	0.617394
7	0.399900	No log	0.712098	0.713714	0.726672	0.713561	0.643696	0.579091	0.618776	0.592137	0.652849
9	0.261500	No log	0.727922	0.716016	0.746756	0.729537	0.630424	0.585796	0.635901	0.597283	0.663410
 [2736/2736 1:20:24, Epoch 17/18]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	1.279600	No log	0.367534	0.474515	0.482856	0.339291	0.143951	0.430584	0.353518	0.123606	0.231449
1	1.035900	No log	0.559469	0.550479	0.622073	0.525183	0.374170	0.473673	0.487131	0.365398	0.445291
3	0.772500	No log	0.656457	0.621253	0.704262	0.640881	0.464523	0.499845	0.534101	0.434867	0.537874
4	0.708300	No log	0.650842	0.617073	0.707545	0.632411	0.503318	0.516408	0.565323	0.468859	0.550635
6	0.414600	No log	0.681981	0.668373	0.714469	0.677682	0.594691	0.554462	0.624706	0.557105	0.617394
7	0.399900	No log	0.712098	0.713714	0.726672	0.713561	0.643696	0.579091	0.618776	0.592137	0.652849
9	0.261500	No log	0.727922	0.716016	0.746756	0.729537	0.630424	0.585796	0.635901	0.597283	0.663410
10	0.233100	No log	0.718224	0.725402	0.731404	0.723807	0.641143	0.579512	0.618299	0.593071	0.658439
12	0.234800	No log	0.729454	0.723231	0.743930	0.732293	0.644717	0.598754	0.633014	0.606822	0.669558
13	0.206000	No log	0.726901	0.731850	0.738112	0.733204	0.652884	0.598919	0.631183	0.610456	0.671830
15	0.170300	No log	0.740684	0.745219	0.748670	0.746687	0.661562	0.610312	0.631037	0.618586	0.682637
16	0.165400	No log	0.741194	0.746656	0.749003	0.747436	0.658499	0.606849	0.630377	0.615617	0.681526
17	0.183500	No log	0.741705	0.747055	0.749379	0.747807	0.661052	0.608669	0.633082	0.617962	0.682885
✅ SECTION 10: Model Training Execution completed in 1.3h 21m
🕒 Total runtime so far: 1.4h 22m
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	xlm_roberta	xlm-roberta-base	0.737755	0.748172	0.722215	0.733459	0.672959	0.613639	0.640367	0.622521	0.67799	4.2005	466.615	23.331


=== Detailed breakdowns for xlm_roberta ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.723780	0.786682	0.753921	886
1	neutral	0.746012	0.702079	0.723379	866
2	positive	0.774725	0.677885	0.723077	208

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.558862	0.672897	0.610601	642
1	objective	0.470852	0.555556	0.509709	189
2	partisan	0.811203	0.692648	0.747253	1129

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.790068	0.644438	0.572165	0.492308	0.868840
1	neutral	866	0.538106	0.518026	0.619266	0.515337	0.419476
2	positive	208	0.735577	0.647529	0.658065	0.476190	0.808333

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	1129	0.775022	0.745098	0.833807	0.649351	0.752137
1	non_polarized	642	0.697819	0.670177	0.528409	0.770000	0.712121
2	objective	189	0.650794	0.545580	0.382022	0.754717	0.500000

Saved detailed breakdowns to: ./runs_xlm_roberta_run13/details

🎯 MULTICLASS CALIBRATION - Optimize prediction biases for better performance
======================================================================

🔧 Calibrating xlm_roberta (xlm-roberta-base)...
📊 Step 1: Extracting polarization logits from trained model...
   Loading model from: ./runs_xlm_roberta_run13/xlm_roberta
   Warning: No trained weights found at ./runs_xlm_roberta_run13/xlm_roberta/pytorch_model.bin, using untrained model
   Loading model from: ./runs_xlm_roberta_run13/xlm_roberta
   Warning: No trained weights found at ./runs_xlm_roberta_run13/xlm_roberta/pytorch_model.bin, using untrained model
   ✓ Validation logits shape: (1959, 3)
   ✓ Test logits shape: (1960, 3)
🔍 Step 2: Searching for optimal bias vector (coordinate search)...
   ✓ Optimal bias vector found (VAL macro-F1=0.379):
      • non_polarized: -0.60
      •     objective: -0.10
      •      partisan: +0.00
📈 Step 3: Evaluating calibration impact on test set...

   📊 TEST MACRO-F1: 0.164 → 0.377 (+0.213)

   Per-class breakdown:
   📉 non_polarized: P=0.328 R=1.000 F1=0.493 (n=642)  →  P=0.375 R=0.419 F1=0.396 (-0.098)
   📈     objective: P=0.000 R=0.000 F1=0.000 (n=189)  →  P=0.161 R=0.101 F1=0.124 (+0.124)
   📈      partisan: P=0.000 R=0.000 F1=0.000 (n=1129)  →  P=0.614 R=0.611 F1=0.613 (+0.613)

✅ Calibration complete! Bias vector saved to:
   ./runs_xlm_roberta_run13/calibration_vector/xlm_roberta_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!

Token indices sequence length is longer than the specified maximum sequence length for this model (950 > 512). Running this sequence through the model will result in indexing errors

[xlm_roberta] xlm-roberta-base
Token length stats: {'mean': 107.0766, 'p50': 96.0, 'p90': 170.0, 'p95': 183.0, 'p99': 223.02000000000044, 'max': 950}
✅ SECTION 11+: Evaluation & Calibration completed in 6.2m 12s
🕒 Total runtime so far: 1.5h 28m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 5.0s
SECTION 3: Configuration Setup           : 0.0s
SECTION 4: Data Loading & Preprocessing  : 40.8s
SECTION 5-9: Model Architecture & Training Setup : 10.7s
SECTION 10: Model Training Execution     : 1.3h 21m
SECTION 11+: Evaluation & Calibration    : 6.2m 12s
======================================== : ==========
TOTAL EXECUTION TIME                     : 1.5h 28m
============================================================
