🚀 Starting SECTION 10: Model Training Execution...

=== Running xlm_roberta -> xlm-roberta-base ===
tokenizer_config.json:   0%|          | 0.00/25.0 [00:00<?, ?B/s]config.json:   0%|          | 0.00/615 [00:00<?, ?B/s]sentencepiece.bpe.model:   0%|          | 0.00/5.07M [00:00<?, ?B/s]tokenizer.json:   0%|          | 0.00/9.10M [00:00<?, ?B/s]model.safetensors:   0%|          | 0.00/1.12G [00:00<?, ?B/s]
🔥 Enhanced Oversampling: min=1.00, max=33.92
   ├─ Objective boosted samples: 405 (target: weak class at 40% F1)
   └─ Neutral boosted samples: 1874 (target: weak class at 49% F1)
 [3190/3190 1:34:16, Epoch 21/22]
Epoch	Training Loss	Validation Loss	Sent Acc	Sent Prec	Sent Rec	Sent F1	Pol Acc	Pol Prec	Pol Rec	Pol F1	Macro F1 Avg
0	1.182100	No log	0.276254	0.300758	0.357296	0.194135	0.062207	0.020736	0.333333	0.039043	0.116589
1	0.916600	No log	0.405351	0.619262	0.504437	0.417920	0.291639	0.477517	0.495070	0.297045	0.357483
3	0.504400	No log	0.396656	0.611541	0.572967	0.421669	0.332441	0.505995	0.532363	0.332552	0.377111
4	0.433800	No log	0.422074	0.656283	0.558722	0.466474	0.558528	0.515488	0.644144	0.494037	0.480256
6	0.192800	No log	0.430769	0.611596	0.604121	0.465451	0.478261	0.559257	0.541990	0.460532	0.462992
7	0.188100	No log	0.494314	0.684394	0.608853	0.547710	0.618060	0.569639	0.622703	0.557215	0.552463
9	0.125500	No log	0.612709	0.660197	0.671386	0.635634	0.679599	0.582847	0.595401	0.569348	0.602491
10	0.133100	No log	0.612040	0.690348	0.657481	0.637919	0.684950	0.589170	0.632289	0.592308	0.615113
12	0.124600	No log	0.592642	0.660235	0.687138	0.620963	0.626087	0.610507	0.581667	0.547268	0.584115
13	0.104900	No log	0.601338	0.663306	0.680876	0.629507	0.650836	0.620048	0.584304	0.558126	0.593817
15	0.108300	No log	0.612709	0.671796	0.685585	0.640528	0.692308	0.620656	0.600330	0.582676	0.611602
16	0.103000	No log	0.610033	0.671296	0.684986	0.638286	0.680268	0.618402	0.597470	0.576046	0.607166
18	0.096300	No log	0.573913	0.671216	0.672823	0.611110	0.632107	0.611019	0.580875	0.548345	0.579728
19	0.083300	No log	0.631438	0.649200	0.693059	0.647176	0.665552	0.614369	0.590033	0.565048	0.606112
21	0.107500	No log	0.599331	0.681629	0.657659	0.628430	0.669565	0.611149	0.598524	0.570874	0.599652
✅ SECTION 10: Model Training Execution completed in 1.6h 35m
🕒 Total runtime so far: 1.6h 36m

=== Detailed breakdowns for xlm_roberta ===

Sentiment — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	negative	0.859459	0.538375	0.662040	886
1	neutral	0.430108	0.798005	0.558952	401
2	positive	0.770408	0.725962	0.747525	208

Polarization — per class (precision/recall/F1/support):
class	precision	recall	f1	support
0	non_polarized	0.494475	0.822989	0.617774	435
1	objective	0.596154	0.344444	0.436620	90
2	partisan	0.872045	0.646392	0.742451	970

Polarity performance within each Sentiment slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_non_polarized	f1_objective	f1_partisan
0	negative	886	0.688488	0.569274	0.498069	0.432432	0.777321
1	neutral	401	0.658354	0.598732	0.734177	0.494118	0.567901
2	positive	208	0.682692	0.533680	0.658683	0.200000	0.742358

Sentiment performance within each Polarity slice (accuracy / macro-F1 / per-class F1):
slice	support	accuracy	macro_f1	f1_negative	f1_neutral	f1_positive
0	partisan	970	0.609278	0.597194	0.698523	0.355805	0.737255
1	non_polarized	435	0.678161	0.678778	0.508333	0.728000	0.800000
2	objective	90	0.688889	0.620364	0.560000	0.774775	0.526316

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
   ✓ Optimal bias vector found (VAL macro-F1=0.302):
      • non_polarized: -0.80
      •     objective: +0.10
      •      partisan: +0.30
📈 Step 3: Evaluating calibration impact on test set...

   📊 TEST MACRO-F1: 0.150 → 0.275 (+0.125)

   Per-class breakdown:
   📉 non_polarized: P=0.291 R=1.000 F1=0.451 (n=435)  →  P=0.000 R=0.000 F1=0.000 (-0.451)
   📈     objective: P=0.000 R=0.000 F1=0.000 (n=90)  →  P=0.182 R=0.022 F1=0.040 (+0.040)
   📈      partisan: P=0.000 R=0.000 F1=0.000 (n=970)  →  P=0.649 R=0.993 F1=0.785 (+0.785)

✅ Calibration complete! Bias vector saved to:
   ./runs_xlm_roberta_optimized/calibration_vector/xlm_roberta_bias_vector.json

======================================================================
🎉 CALIBRATION FINISHED - All models optimized!


[xlm_roberta] xlm-roberta-base
Token indices sequence length is longer than the specified maximum sequence length for this model (950 > 512). Running this sequence through the model will result in indexing errors
Token length stats: {'mean': 106.9514, 'p50': 96.0, 'p90': 170.0, 'p95': 182.0, 'p99': 215.0, 'max': 950}
✅ SECTION 11+: Evaluation & Calibration completed in 25.2s
🕒 Total runtime so far: 1.6h 36m
------------------------------------------------------------

============================================================
⏱️  EXECUTION TIME SUMMARY
============================================================
SECTION 2: Environment & Imports         : 10.7s
SECTION 3: Configuration Setup           : 1.1s
SECTION 4: Data Loading & Preprocessing  : 3.6s
SECTION 5-9: Model Architecture & Training Setup : 8.6s
SECTION 10: Model Training Execution     : 1.6h 35m
SECTION 11+: Evaluation & Calibration    : 25.2s
======================================== : ==========
TOTAL EXECUTION TIME                     : 1.6h 36m
============================================================