🚀 Starting SECTION 10: Model Training Execution...

=== Running mbert -> bert-base-multilingual-cased ===
tokenizer_config.json:   0%|          | 0.00/49.0 [00:00<?, ?B/s]config.json:   0%|          | 0.00/625 [00:00<?, ?B/s]vocab.txt:   0%|          | 0.00/996k [00:00<?, ?B/s]tokenizer.json:   0%|          | 0.00/1.96M [00:00<?, ?B/s]model.safetensors:   0%|          | 0.00/714M [00:00<?, ?B/s]
🔥 Enhanced Oversampling: min=1.00, max=1.90
   ├─ Objective boosted samples: 0 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 0 (target: weak class at 49% F1)
 [9240/9240 1:51:28, Epoch 19/20]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	0.926200	No log	0.546197	0.506952	0.548536	0.512503	0.303726	0.461782	0.456537	0.299731	0.406117
1	0.700400	No log	0.602348	0.570261	0.647913	0.577088	0.522716	0.522385	0.592525	0.502893	0.539990
2	0.521500	No log	0.659520	0.628536	0.715731	0.643866	0.581419	0.558271	0.643754	0.547414	0.595640
4	0.323500	No log	0.765697	0.761321	0.810941	0.780175	0.692190	0.659397	0.744318	0.679076	0.729626
5	0.278400	No log	0.787136	0.792629	0.833251	0.807168	0.660541	0.673669	0.736925	0.674839	0.741003
6	0.212400	No log	0.834099	0.853900	0.866569	0.859920	0.750383	0.712078	0.794857	0.740862	0.800391
8	0.171900	No log	0.835630	0.875297	0.877118	0.868019	0.757529	0.738177	0.822718	0.761384	0.814702
9	0.127100	No log	0.887187	0.901888	0.916047	0.908506	0.795304	0.764732	0.839434	0.791787	0.850147
10	0.122100	No log	0.898928	0.922606	0.925065	0.922002	0.792241	0.761760	0.847846	0.789687	0.855845
12	0.082800	No log	0.920368	0.935929	0.941023	0.936813	0.805513	0.783381	0.858013	0.806157	0.871485
13	0.090300	No log	0.932619	0.950161	0.950165	0.948437	0.815212	0.790422	0.871984	0.816698	0.882567
14	0.097700	No log	0.951506	0.960554	0.964007	0.961988	0.821337	0.798670	0.875972	0.823888	0.892938
16	0.066700	No log	0.974987	0.978938	0.981417	0.980122	0.831547	0.805145	0.883105	0.831384	0.905753
17	0.072700	No log	0.981113	0.984712	0.984672	0.984688	0.845329	0.811027	0.886297	0.839779	0.912234
18	0.056100	No log	0.984176	0.986968	0.988215	0.987588	0.843798	0.812123	0.888270	0.839974	0.913781
19	0.051100	No log	0.985707	0.988121	0.989344	0.988730	0.843798	0.813105	0.889240	0.840580	0.914655
✅ SECTION 10: Model Training Execution completed in 1.9h 52m
🕒 Total runtime so far: 1.9h 55m
------------------------------------------------------------

🚀 Starting SECTION 11+: Evaluation & Calibration...
model_key	base_name	test_test_sent_acc	test_test_sent_prec	test_test_sent_rec	test_test_sent_f1	test_test_pol_acc	test_test_pol_prec	test_test_pol_rec	test_test_pol_f1	test_test_macro_f1_avg	test_test_runtime	test_test_samples_per_second	test_test_steps_per_second
0	mbert	bert-base-multilingual-cased	0.981633	0.983863	0.986341	0.985077	0.845918	0.820192	0.893018	0.846428	0.915752	4.2996	455.861	38.143

=== Detailed breakdowns for mbert ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.986301	0.975169	0.980704	886
1	neutral	0.974857	0.983852	0.979334	867
2	positive	0.990431	1.000000	0.995192	207

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.727031	0.882068	0.797080	619
1	objective	0.768953	1.000000	0.869388	213
2	partisan	0.964592	0.796986	0.872816	1128

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.946953	0.931390	0.844884	0.981818	0.967468
1	neutral	867	0.712803	0.685240	0.760893	0.847458	0.447368
2	positive	207	0.971014	0.976608	0.952381	1.000000	0.977444

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	1128	0.991135	0.991674	0.993007	0.982014	1.000000
1	non_polarized	619	0.967690	0.967909	0.927536	0.976190	1.000000
2	objective	213	0.971831	0.942632	0.928571	0.982659	0.916667

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
   ✓ Optimal bias vector found (VAL macro-F1=0.845):
      • non_polarized: -0.20
      •     objective: -0.30
      •      partisan: +0.00
📈 Step 3: Evaluating calibration impact on test set...

   📊 TEST MACRO-F1: 0.846 → 0.849 (+0.003)

   Per-class breakdown:
   📉 non_polarized: P=0.726 R=0.882 F1=0.796 (n=619)  →  P=0.777 R=0.806 F1=0.791 (-0.005)
   📈     objective: P=0.769 R=1.000 F1=0.869 (n=213)  →  P=0.772 R=1.000 F1=0.871 (+0.002)
   📈      partisan: P=0.965 R=0.796 F1=0.872 (n=1128)  →  P=0.922 R=0.852 F1=0.886 (+0.013)

✅ Calibration complete! Bias vector saved to:
   ./runs_mbert_optimized/calibration_vector/mbert_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!

[mbert] bert-base-multilingual-cased
Token indices sequence length is longer than the specified maximum sequence length for this model (916 > 512). Running this sequence through the model will result in indexing errors
Token length stats: {'mean': 109.236, 'p50': 98.0, 'p90': 181.0, 'p95': 195.0, 'p99': 226.0, 'max': 916}
✅ SECTION 11+: Evaluation & Calibration completed in 16.5s
🕒 Total runtime so far: 1.9h 55m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 4.8s
SECTION 3: Configuration Setup           : 2.7m 41s
SECTION 4: Data Loading & Preprocessing  : 3.4s
SECTION 5-9: Model Architecture & Training Setup : 8.2s
SECTION 10: Model Training Execution     : 1.9h 52m
SECTION 11+: Evaluation & Calibration    : 16.5s
======================================== : ==========
TOTAL EXECUTION TIME                     : 1.9h 55m
============================================================