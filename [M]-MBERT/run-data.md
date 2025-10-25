🚀 Starting SECTION 10: Model Training Execution...

=== Running mbert -> bert-base-multilingual-cased ===
🔥 Enhanced Oversampling: min=1.00, max=68.28
   ├─ Objective boosted samples: 405 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 1874 (target: weak class at 49% F1)
 [2900/2900 1:30:46, Epoch 19/20]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	0.981400	No log	0.284281	0.291616	0.379744	0.229431	0.062876	0.131889	0.334179	0.040796	0.135114
1	0.757100	No log	0.367893	0.623152	0.489539	0.381896	0.168562	0.485688	0.400857	0.162680	0.272288
3	0.323700	No log	0.406689	0.627172	0.514156	0.423780	0.321070	0.489213	0.540787	0.315410	0.369595
4	0.198600	No log	0.486288	0.660490	0.575533	0.528005	0.511037	0.508579	0.607258	0.476934	0.502470
6	0.163300	No log	0.447492	0.669128	0.563225	0.491387	0.608696	0.539851	0.612374	0.540954	0.516170
7	0.143100	No log	0.494983	0.645094	0.609964	0.536409	0.493645	0.549756	0.547019	0.467266	0.501837
9	0.131600	No log	0.484950	0.675598	0.578779	0.524403	0.660870	0.576606	0.609916	0.575482	0.549942
10	0.111600	No log	0.557860	0.624804	0.655765	0.582300	0.709030	0.590974	0.590853	0.586532	0.584416
12	0.078900	No log	0.578595	0.670773	0.634186	0.608496	0.713712	0.592281	0.609627	0.596433	0.602465
13	0.077000	No log	0.587960	0.643669	0.652857	0.612647	0.632107	0.578982	0.588705	0.556034	0.584340
15	0.068400	No log	0.597324	0.642275	0.656890	0.617900	0.625418	0.584734	0.577022	0.545511	0.581706
16	0.060300	No log	0.566555	0.650996	0.648518	0.601008	0.593980	0.592310	0.568179	0.530394	0.565701
18	0.064100	No log	0.620736	0.659349	0.667419	0.641035	0.674247	0.607198	0.590694	0.574767	0.607901
19	0.059000	No log	0.601338	0.668899	0.667051	0.631438	0.673579	0.599349	0.581279	0.566972	0.599205
✅ SECTION 10: Model Training Execution completed in 1.5h 32m
🕒 Total runtime so far: 4.6h 37m
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	mbert	bert-base-multilingual-cased	0.628763	0.661554	0.671064	0.643388	0.681605	0.620283	0.602279	0.588496	0.615942	5.1248	291.718	18.342

=== Detailed breakdowns for mbert ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.831933	0.558691	0.668467	886
1	neutral	0.425899	0.738155	0.540146	401
2	positive	0.726829	0.716346	0.721550	208

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.504225	0.822989	0.625328	435
1	objective	0.483871	0.333333	0.394737	90
2	partisan	0.872752	0.650515	0.745422	970

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.697517	0.515252	0.505133	0.250000	0.790622
1	neutral	401	0.648379	0.593558	0.731405	0.530120	0.519149
2	positive	208	0.677885	0.530058	0.666667	0.190476	0.733032

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	970	0.637113	0.607066	0.730318	0.357290	0.733591
1	non_polarized	435	0.606897	0.593030	0.378855	0.678431	0.721805
2	objective	90	0.644444	0.610678	0.533333	0.727273	0.571429

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

[mbert] bert-base-multilingual-cased
Token indices sequence length is longer than the specified maximum sequence length for this model (916 > 512). Running this sequence through the model will result in indexing errors
Token length stats: {'mean': 109.174, 'p50': 97.0, 'p90': 179.0, 'p95': 194.0, 'p99': 226.02000000000044, 'max': 916}
✅ SECTION 11+: Evaluation & Calibration completed in 13.2s
🕒 Total runtime so far: 4.6h 37m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 18.3s
SECTION 3: Configuration Setup           : 16.7m 40s
SECTION 4: Data Loading & Preprocessing  : 3.4m 21s
SECTION 5-9: Model Architecture & Training Setup : 10.3s
SECTION 10: Model Training Execution     : 1.5h 32m
SECTION 11+: Evaluation & Calibration    : 13.2s
======================================== : ==========
TOTAL EXECUTION TIME                     : 4.6h 37m
============================================================