
🚀 Starting SECTION 10: Model Training Execution...

=== Running mbert -> bert-base-multilingual-cased ===
tokenizer_config.json:   0%|          | 0.00/49.0 [00:00<?, ?B/s]config.json:   0%|          | 0.00/625 [00:00<?, ?B/s]vocab.txt:   0%|          | 0.00/996k [00:00<?, ?B/s]tokenizer.json:   0%|          | 0.00/1.96M [00:00<?, ?B/s]model.safetensors:   0%|          | 0.00/714M [00:00<?, ?B/s]
🔥 Enhanced Oversampling: min=1.00, max=68.28
   ├─ Objective boosted samples: 405 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 1874 (target: weak class at 49% F1)
 [2900/2900 1:02:43, Epoch 19/20]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	0.978500	No log	0.286288	0.295020	0.384575	0.235865	0.063545	0.087542	0.335025	0.042591	0.139228
1	0.686600	No log	0.408027	0.568599	0.535749	0.429385	0.131104	0.502719	0.381308	0.129856	0.279621
3	0.348000	No log	0.436789	0.610607	0.555019	0.468894	0.408027	0.492307	0.585260	0.389882	0.429388
4	0.201300	No log	0.443478	0.640605	0.575402	0.488142	0.524415	0.515626	0.616772	0.490541	0.489341
6	0.159900	No log	0.466890	0.661825	0.589023	0.515630	0.691639	0.575579	0.626643	0.589953	0.552792
7	0.147200	No log	0.513043	0.631211	0.632744	0.551362	0.521739	0.544921	0.553886	0.477601	0.514482
9	0.142000	No log	0.507692	0.658965	0.630812	0.552155	0.634783	0.565698	0.615540	0.560976	0.556565
10	0.123000	No log	0.527090	0.596961	0.643224	0.550629	0.685619	0.592439	0.602630	0.582847	0.566738
12	0.091900	No log	0.577926	0.657321	0.639980	0.605339	0.719732	0.594277	0.602489	0.594551	0.599945
13	0.076400	No log	0.555853	0.652044	0.644889	0.592473	0.646154	0.585070	0.590011	0.557650	0.575062
15	0.072200	No log	0.584615	0.634871	0.652289	0.609825	0.652843	0.574443	0.583719	0.552774	0.581300
16	0.070500	No log	0.533779	0.649557	0.631614	0.573111	0.570569	0.578493	0.544137	0.493990	0.533550
18	0.084400	No log	0.598662	0.655780	0.656862	0.623720	0.653512	0.573664	0.566751	0.542620	0.583170
19	0.072400	No log	0.586622	0.653203	0.655604	0.616206	0.660870	0.579149	0.570550	0.550118	0.583162
✅ SECTION 10: Model Training Execution completed in 1.1h 3m
🕒 Total runtime so far: 1.1h 7m
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	mbert	bert-base-multilingual-cased	0.590635	0.658584	0.650413	0.614125	0.735786	0.629745	0.632101	0.627129	0.620627	4.7061	317.676	19.974




=== Detailed breakdowns for mbert ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.847082	0.475169	0.608821	886
1	neutral	0.399504	0.802993	0.533554	401
2	positive	0.729167	0.673077	0.700000	208

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.582255	0.724138	0.645492	435
1	objective	0.450000	0.400000	0.423529	90
2	partisan	0.856979	0.772165	0.812364	970

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.782167	0.598866	0.572165	0.363636	0.860798
1	neutral	401	0.650873	0.608349	0.700935	0.500000	0.624113
2	positive	208	0.701923	0.566606	0.675000	0.260870	0.763948

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	970	0.579381	0.573187	0.667265	0.356643	0.695652
1	non_polarized	435	0.600000	0.578497	0.317757	0.681733	0.736000
2	objective	90	0.666667	0.617003	0.555556	0.750000	0.545455


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
   ✓ Optimal bias vector found (VAL macro-F1=0.339):
      • non_polarized: -0.10
      •     objective: +0.20
      •      partisan: +0.00
📈 Step 3: Evaluating calibration impact on test set...

   📊 TEST MACRO-F1: 0.292 → 0.285 (-0.007)

   Per-class breakdown:
   📉 non_polarized: P=0.273 R=0.007 F1=0.013 (n=435)  →  P=0.000 R=0.000 F1=0.000 (-0.013)
   📈     objective: P=0.095 R=0.133 F1=0.111 (n=90)  →  P=0.106 R=0.511 F1=0.176 (+0.065)
   📉      partisan: P=0.644 R=0.901 F1=0.751 (n=970)  →  P=0.649 R=0.710 F1=0.678 (-0.073)

✅ Calibration complete! Bias vector saved to:
   ./runs_mbert_optimized/calibration_vector/mbert_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!
