🚀 Starting SECTION 10: Model Training Execution...

=== Running xlm_roberta -> xlm-roberta-base ===
tokenizer_config.json:   0%|          | 0.00/25.0 [00:00<?, ?B/s]config.json:   0%|          | 0.00/615 [00:00<?, ?B/s]sentencepiece.bpe.model:   0%|          | 0.00/5.07M [00:00<?, ?B/s]tokenizer.json:   0%|          | 0.00/9.10M [00:00<?, ?B/s]model.safetensors:   0%|          | 0.00/1.12G [00:00<?, ?B/s]
🔥 Enhanced Oversampling: min=1.00, max=4.43
   ├─ Objective boosted samples: 168 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 0 (target: weak class at 49% F1)
 [2736/2736 2:07:14, Epoch 17/18]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	1.157400	No log	0.393058	0.489435	0.498137	0.333372	0.159775	0.466084	0.365538	0.146513	0.239942
1	0.880300	No log	0.543645	0.546480	0.615181	0.526503	0.314957	0.493104	0.456442	0.309705	0.418104
3	0.661100	No log	0.625319	0.604194	0.685727	0.606802	0.423686	0.496401	0.537297	0.416652	0.511727
4	0.558800	No log	0.664625	0.637311	0.703859	0.650563	0.502808	0.521407	0.575998	0.485836	0.568199
6	0.363700	No log	0.636039	0.644639	0.690898	0.622831	0.600817	0.552891	0.615796	0.559128	0.590980
7	0.352400	No log	0.650842	0.625013	0.710926	0.628460	0.623788	0.579000	0.614328	0.586415	0.607438
9	0.237200	No log	0.715161	0.696345	0.747510	0.713930	0.626340	0.564634	0.619621	0.573144	0.643537
10	0.222600	No log	0.733027	0.724924	0.749614	0.735050	0.632976	0.606347	0.621419	0.602018	0.668534
12	0.191200	No log	0.742215	0.750819	0.742316	0.745834	0.639102	0.604626	0.628512	0.606606	0.676220
13	0.199400	No log	0.747831	0.755115	0.753381	0.752353	0.626340	0.599670	0.631513	0.600586	0.676469
15	0.155700	No log	0.753956	0.756263	0.755141	0.755585	0.658499	0.608602	0.635409	0.618473	0.687029
16	0.152700	No log	0.750383	0.754109	0.750039	0.751873	0.658499	0.611311	0.638442	0.620628	0.686250
17	0.150000	No log	0.754467	0.757946	0.754404	0.755693	0.656457	0.612036	0.639301	0.620503	0.688098
✅ SECTION 10: Model Training Execution completed in 2.1h 9m
🕒 Total runtime so far: 2.1h 9m
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	xlm_roberta	xlm-roberta-base	0.741837	0.746566	0.724598	0.734148	0.660714	0.607518	0.639577	0.615402	0.674775	7.0357	278.581	13.929

=== Detailed breakdowns for xlm_roberta ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.761962	0.718962	0.739837	886
1	neutral	0.720980	0.781755	0.750139	866
2	positive	0.756757	0.673077	0.712468	208

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.541818	0.696262	0.609407	642
1	objective	0.459227	0.566138	0.507109	189
2	partisan	0.821508	0.656333	0.729690	1129

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.782167	0.645676	0.586538	0.491803	0.858687
1	neutral	866	0.523095	0.498068	0.614173	0.508772	0.371257
2	positive	208	0.716346	0.650423	0.641975	0.526316	0.782979

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	1129	0.755536	0.734060	0.809344	0.660920	0.731915
1	non_polarized	642	0.728972	0.682924	0.529968	0.807692	0.711111
2	objective	189	0.703704	0.577813	0.410256	0.801444	0.521739

Saved detailed breakdowns to: ./runs_xlm_roberta_optimized/details

🎯 MULTICLASS CALIBRATION - Optimize prediction biases for better performance
======================================================================

🔧 Calibrating xlm_roberta (xlm-roberta-base)...
📊 Step 1: Extracting polarization logits from trained model...
   Loading model from: ./runs_xlm_roberta_optimized/xlm_roberta
   Warning: No trained weights found at ./runs_xlm_roberta_optimized/xlm_roberta/pytorch_model.bin, using untrained model
   Loading model from: ./runs_xlm_roberta_optimized/xlm_roberta
   Warning: No trained weights found at ./runs_xlm_roberta_optimized/xlm_roberta/pytorch_model.bin, using untrained model
   ✓ Validation logits shape: (1959, 3)
   ✓ Test logits shape: (1960, 3)
🔍 Step 2: Searching for optimal bias vector (coordinate search)...
   ✓ Optimal bias vector found (VAL macro-F1=0.308):
      • non_polarized: +0.80
      •     objective: -0.80
      •      partisan: +0.20
📈 Step 3: Evaluating calibration impact on test set...

   📊 TEST MACRO-F1: 0.164 → 0.164 (+0.000)

   Per-class breakdown:
   ➡️ non_polarized: P=0.328 R=1.000 F1=0.493 (n=642)  →  P=0.328 R=1.000 F1=0.493 (+0.000)
   ➡️     objective: P=0.000 R=0.000 F1=0.000 (n=189)  →  P=0.000 R=0.000 F1=0.000 (+0.000)
   ➡️      partisan: P=0.000 R=0.000 F1=0.000 (n=1129)  →  P=0.000 R=0.000 F1=0.000 (+0.000)

✅ Calibration complete! Bias vector saved to:
   ./runs_xlm_roberta_optimized/calibration_vector/xlm_roberta_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!

Token indices sequence length is longer than the specified maximum sequence length for this model (950 > 512). Running this sequence through the model will result in indexing errors

[xlm_roberta] xlm-roberta-base
Token length stats: {'mean': 107.0766, 'p50': 96.0, 'p90': 170.0, 'p95': 183.0, 'p99': 223.02000000000044, 'max': 950}
✅ SECTION 11+: Evaluation & Calibration completed in 25.2s
🕒 Total runtime so far: 2.2h 9m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 16.3s
SECTION 3: Configuration Setup           : 0.1s
SECTION 4: Data Loading & Preprocessing  : 0.3s
SECTION 5-9: Model Architecture & Training Setup : 0.2s
SECTION 10: Model Training Execution     : 2.1h 9m
SECTION 11+: Evaluation & Calibration    : 25.2s
======================================== : ==========
TOTAL EXECUTION TIME                     : 2.2h 9m
============================================================