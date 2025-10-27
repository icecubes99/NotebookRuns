🚀 Starting SECTION 10: Model Training Execution...

=== Running mbert -> bert-base-multilingual-cased ===
tokenizer_config.json:   0%|          | 0.00/49.0 [00:00<?, ?B/s]config.json:   0%|          | 0.00/625 [00:00<?, ?B/s]vocab.txt:   0%|          | 0.00/996k [00:00<?, ?B/s]tokenizer.json:   0%|          | 0.00/1.96M [00:00<?, ?B/s]model.safetensors:   0%|          | 0.00/714M [00:00<?, ?B/s]
🔥 Enhanced Oversampling: min=1.00, max=1.90
   ├─ Objective boosted samples: 0 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 0 (target: weak class at 49% F1)
 [9240/9240 3:24:26, Epoch 19/20]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	0.880600	No log	0.559980	0.525459	0.582237	0.535779	0.303216	0.473146	0.456775	0.299965	0.417872
1	0.687000	No log	0.607963	0.575826	0.656571	0.583816	0.552833	0.526687	0.599516	0.521384	0.552600
2	0.536900	No log	0.590097	0.591617	0.684023	0.569663	0.537519	0.538011	0.625820	0.512161	0.540912
4	0.330500	No log	0.753956	0.764765	0.804791	0.779451	0.684533	0.639755	0.731589	0.663996	0.721723
5	0.300200	No log	0.786115	0.793641	0.834455	0.811634	0.692190	0.682378	0.758920	0.696624	0.754129
6	0.210100	No log	0.812149	0.852623	0.854728	0.852791	0.741194	0.699684	0.777890	0.726955	0.789873
8	0.169300	No log	0.830526	0.868735	0.873295	0.863507	0.719755	0.724748	0.803721	0.735492	0.799499
9	0.146600	No log	0.874426	0.899695	0.901687	0.900195	0.779990	0.748150	0.824007	0.776015	0.838105
10	0.154900	No log	0.884635	0.910099	0.913227	0.909428	0.771822	0.753171	0.831685	0.776062	0.842745
12	0.100400	No log	0.929556	0.944406	0.946456	0.944998	0.788157	0.770151	0.845038	0.791937	0.868467
13	0.100800	No log	0.925472	0.942663	0.944863	0.941830	0.790710	0.773458	0.858210	0.795203	0.868516
14	0.078400	No log	0.942828	0.955397	0.957689	0.955477	0.813680	0.787632	0.870368	0.814120	0.884798
16	0.078900	No log	0.966309	0.972638	0.975004	0.973649	0.832057	0.804258	0.884420	0.830874	0.902261
17	0.052200	No log	0.970904	0.975983	0.978408	0.977076	0.829505	0.802817	0.884637	0.828694	0.902885
18	0.061400	No log	0.978050	0.981187	0.983675	0.982406	0.840225	0.809070	0.889399	0.836589	0.909497
19	0.066300	No log	0.977029	0.980446	0.982931	0.981645	0.845329	0.812770	0.892117	0.840727	0.911186
✅ SECTION 10: Model Training Execution completed in 3.4h 26m
🕒 Total runtime so far: 3.5h 28m
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	mbert	bert-base-multilingual-cased	0.978571	0.980454	0.9841	0.982187	0.839796	0.812217	0.890444	0.838918	0.910552	7.0353	278.594	23.311

=== Detailed breakdowns for mbert ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.988453	0.966140	0.977169	886
1	neutral	0.967195	0.986159	0.976585	867
2	positive	0.985714	1.000000	0.992806	207

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.718954	0.888530	0.794798	619
1	objective	0.747368	1.000000	0.855422	213
2	partisan	0.970330	0.782801	0.866536	1128

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.942438	0.918067	0.842444	0.947368	0.964387
1	neutral	867	0.705882	0.672244	0.759788	0.835322	0.421622
2	positive	207	0.961353	0.969066	0.937500	1.000000	0.969697

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	1128	0.984043	0.984485	0.987342	0.969805	0.996310
1	non_polarized	619	0.969305	0.969439	0.930909	0.977408	1.000000
2	objective	213	0.976526	0.949237	0.945455	0.985591	0.916667

Saved detailed breakdowns to: ./runs_mbert_optimized/details

🎯 MULTICLASS CALIBRATION - Optimize prediction biases for better performance
======================================================================

🔧 Calibrating mbert (bert-base-multilingual-cased)...
📊 Step 1: Extracting polarization logits from trained model...
   Loading model from: ./runs_mbert_optimized/mbert
   ✓ Loading weights from: ./runs_mbert_optimized/mbert/model.safetensors
   Loading model from: ./runs_mbert_optimized/mbert
   ✓ Loading weights from: ./runs_mbert_optimized/mbert/model.safetensors
   ✓ Validation logits shape: (1959, 3)
   ✓ Test logits shape: (1960, 3)
🔍 Step 2: Searching for optimal bias vector (coordinate search)...
   ✓ Optimal bias vector found (VAL macro-F1=0.842):
      • non_polarized: +0.00
      •     objective: -0.20
      •      partisan: +0.00
📈 Step 3: Evaluating calibration impact on test set...

   📊 TEST MACRO-F1: 0.839 → 0.840 (+0.001)

   Per-class breakdown:
   📈 non_polarized: P=0.719 R=0.889 F1=0.795 (n=619)  →  P=0.718 R=0.890 F1=0.795 (+0.000)
   📈     objective: P=0.747 R=1.000 F1=0.855 (n=213)  →  P=0.752 R=0.995 F1=0.857 (+0.001)
   📈      partisan: P=0.970 R=0.783 F1=0.867 (n=1128)  →  P=0.970 R=0.784 F1=0.867 (+0.001)

✅ Calibration complete! Bias vector saved to:
   ./runs_mbert_optimized/calibration_vector/mbert_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!

[mbert] bert-base-multilingual-cased
Token indices sequence length is longer than the specified maximum sequence length for this model (916 > 512). Running this sequence through the model will result in indexing errors
Token length stats: {'mean': 109.236, 'p50': 98.0, 'p90': 181.0, 'p95': 195.0, 'p99': 226.0, 'max': 916}
✅ SECTION 11+: Evaluation & Calibration completed in 22.2s
🕒 Total runtime so far: 3.5h 29m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 9.1s
SECTION 3: Configuration Setup           : 19.7s
SECTION 4: Data Loading & Preprocessing  : 14.6s
SECTION 5-9: Model Architecture & Training Setup : 46.2s
SECTION 10: Model Training Execution     : 3.4h 26m
SECTION 11+: Evaluation & Calibration    : 22.2s
======================================== : ==========
TOTAL EXECUTION TIME                     : 3.5h 29m
