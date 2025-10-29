🚀 Starting SECTION 10: Model Training Execution...

=== Running xlm_roberta -> xlm-roberta-base ===
 [6660/6660 2:27:20, Epoch 17/18]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	0.884200	No log	0.557938	0.535255	0.608846	0.534919	0.491577	0.517667	0.543007	0.438539	0.486729
1	0.721900	No log	0.631955	0.600218	0.684700	0.606315	0.555385	0.538179	0.611331	0.524269	0.565292
3	0.505200	No log	0.734048	0.700730	0.785651	0.726391	0.655436	0.614250	0.694188	0.630054	0.678223
4	0.417300	No log	0.784584	0.787923	0.815258	0.798124	0.726901	0.670250	0.752234	0.689250	0.743687
6	0.292600	No log	0.824400	0.838028	0.867807	0.851688	0.758040	0.713008	0.805753	0.742633	0.797160
7	0.236300	No log	0.842266	0.866521	0.878359	0.870961	0.773864	0.731376	0.807414	0.754463	0.812712
9	0.188700	No log	0.886166	0.905700	0.914043	0.909564	0.795304	0.765464	0.848072	0.794943	0.852254
10	0.128000	No log	0.905054	0.929506	0.929614	0.927855	0.820316	0.789490	0.861025	0.817390	0.872622
12	0.108400	No log	0.934661	0.949435	0.951479	0.950040	0.847882	0.809408	0.879534	0.836614	0.893327
13	0.107000	No log	0.944359	0.957987	0.958722	0.957892	0.848392	0.814331	0.888267	0.842464	0.900178
15	0.085100	No log	0.955590	0.965787	0.966991	0.966273	0.859622	0.823961	0.895071	0.851955	0.909114
16	0.088200	No log	0.957631	0.967343	0.968522	0.967795	0.863196	0.826835	0.894476	0.853901	0.910848
17	0.082800	No log	0.956100	0.966184	0.967375	0.966654	0.861154	0.824824	0.893001	0.852047	0.909350
✅ SECTION 10: Model Training Execution completed in 2.5h 30m
🕒 Total runtime so far: 5.4h 26m
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	xlm_roberta	xlm-roberta-base	0.915816	0.918235	0.920156	0.919143	0.885204	0.882145	0.887850	0.884920	0.902032	4.423	443.137	22.157

=== Detailed breakdowns for xlm_roberta ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.912340	0.918345	0.915332	886
1	neutral	0.920125	0.915235	0.917673	867
2	positive	0.922240	0.927295	0.924760	207

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.878125	0.885420	0.881758	619
1	objective	0.882235	0.893145	0.887655	213
2	partisan	0.886075	0.885985	0.886030	1128

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.968397	0.961066	0.902778	1.000000	0.980420
1	neutral	867	0.746251	0.740539	0.774487	0.857843	0.589286
2	positive	207	0.971014	0.975980	0.950000	1.000000	0.977941

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	1128	0.976064	0.977445	0.981158	0.951175	1.000000
1	non_polarized	619	0.957997	0.958483	0.906475	0.968974	1.000000
2	objective	213	0.915493	0.828571	0.619048	0.950000	0.916667

Saved detailed breakdowns to: ./runs_xlm_roberta_run16/details

🎯 MULTICLASS CALIBRATION - Optimize prediction biases for better performance
======================================================================

🔧 Calibrating xlm_roberta (xlm-roberta-base)...
📊 Step 1: Extracting polarization logits from trained model...
   Loading model from: ./runs_xlm_roberta_run16/xlm_roberta
   Loading model from: ./runs_xlm_roberta_run16/xlm_roberta
   ✓ Validation logits shape: (1959, 3)
   ✓ Test logits shape: (1960, 3)
🔍 Step 2: Searching for optimal bias vector (coordinate search)...
   ✓ Optimal bias vector found (VAL macro-F1=0.854):
      • non_polarized: +0.00
      •     objective: +0.00
      •      partisan: -0.10
📈 Step 3: Evaluating calibration impact on test set...

   ⚠️ TEST MACRO-F1: 0.867 → 0.866 (-0.001) < threshold (+0.002); keeping raw logits.

   Per-class breakdown:
   ➡️ non_polarized: P=0.791 R=0.851 F1=0.820 (n=619)  →  P=0.791 R=0.851 F1=0.820 (+0.000)
   ➡️     objective: P=0.786 R=1.000 F1=0.880 (n=213)  →  P=0.786 R=1.000 F1=0.880 (+0.000)
   ➡️      partisan: P=0.945 R=0.857 F1=0.899 (n=1128)  →  P=0.945 R=0.857 F1=0.899 (+0.000)

✅ Calibration complete! Bias vector saved to:
   ./runs_xlm_roberta_run16/calibration_vector/xlm_roberta_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!

[xlm_roberta] xlm-roberta-base
Token indices sequence length is longer than the specified maximum sequence length for this model (950 > 512). Running this sequence through the model will result in indexing errors
Token length stats: {'mean': 106.743, 'p50': 96.0, 'p90': 170.0, 'p95': 182.0, 'p99': 215.0, 'max': 950}
✅ SECTION 11+: Evaluation & Calibration completed in 22.2s
🕒 Total runtime so far: 5.4h 27m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 5.2s
SECTION 3: Configuration Setup           : 25.8m 49s
SECTION 4: Data Loading & Preprocessing  : 21.9s
SECTION 5-9: Model Architecture & Training Setup : 21.1s
SECTION 10: Model Training Execution     : 2.5h 30m
SECTION 11+: Evaluation & Calibration    : 22.2s
======================================== : ==========
TOTAL EXECUTION TIME                     : 5.4h 27m
============================================================