======================================================================
🚀 STARTING AUGMENTATION PROCESS
======================================================================
📦 Loading XLM-RoBERTa for contextual augmentation...
✅ XLM-RoBERTa ready (understands Filipino + Taglish!)
📦 Loading sentence transformer for quality filtering...

The following layers were not sharded: encoder.layer.*.output.LayerNorm.bias, embeddings.position_embeddings.weight, encoder.layer.*.output.dense.bias, encoder.layer.*.intermediate.dense.bias, embeddings.word_embeddings.weight, encoder.layer.*.attention.self.key.bias, encoder.layer.*.output.LayerNorm.weight, encoder.layer.*.attention.output.dense.weight, encoder.layer.*.attention.self.value.weight, encoder.layer.*.output.dense.weight, embeddings.token_type_embeddings.weight, pooler.dense.weight, encoder.layer.*.intermediate.dense.weight, encoder.layer.*.attention.self.query.bias, encoder.layer.*.attention.self.value.bias, pooler.dense.bias, embeddings.LayerNorm.weight, encoder.layer.*.attention.output.LayerNorm.weight, embeddings.LayerNorm.bias, encoder.layer.*.attention.self.query.weight, encoder.layer.*.attention.self.key.weight, encoder.layer.*.attention.output.dense.bias, encoder.layer.*.attention.output.LayerNorm.bias

✅ Quality filter ready (threshold: 0.7)
\n======================================================================
🎯 PHASE 1: AUGMENTING OBJECTIVE CLASS
======================================================================
\n📊 Original objective samples: 588
🎯 Target: ~2940 samples (5x)
🔄 Augmenting 588 samples (x4 each)...

XLM-R augmentation: 100%
 588/588 [00:01<00:00, 323.31it/s]

✅ Generated 2352 samples via XLM-RoBERTa
\n📊 Generated 2352 augmented samples
\n🔍 Applying quality filter...
Encoding 588 original texts...

Batches: 100%
 19/19 [00:02<00:00, 15.42it/s]

Encoding 2352 augmented texts...

Batches: 100%
 74/74 [00:08<00:00, 26.91it/s]
Quality filtering: 100%
 2352/2352 [00:01<00:00, 2390.84it/s]

✅ Kept 2352/2352 (100.0% quality rate)
\n🔍 Removing duplicates...
Encoding 2352 texts for duplicate detection...

Batches: 100%
 74/74 [00:08<00:00, 27.25it/s]
Duplicate removal: 100%
 2351/2351 [00:00<00:00, 2885.25it/s]

✅ Kept 585/2352 unique samples (removed 1767 duplicates)
\n📝 Mapping titles to augmented samples...
✅ Assigned 36 unique titles to augmented samples
\n======================================================================
✅ OBJECTIVE CLASS COMPLETE!
======================================================================
📊 Original: 588
📊 Augmented: 585
📊 Total: 1173
📊 Multiplier: 1.99x
\n======================================================================
🎯 PHASE 2: AUGMENTING NEUTRAL CLASS
======================================================================
\n📊 Original neutral samples: 2677
🎯 Target: ~8031 samples (3x)
🔄 Augmenting 2677 samples (x2 each)...

XLM-R augmentation: 100%
 2677/2677 [00:03<00:00, 813.96it/s]

✅ Generated 5354 samples via XLM-RoBERTa
\n📊 Generated 5354 augmented samples
\n🔍 Applying quality filter...
Encoding 2677 original texts...

Batches: 100%
 84/84 [00:05<00:00, 36.93it/s]

Encoding 5354 augmented texts...

Batches: 100%
 168/168 [00:10<00:00, 42.66it/s]
Quality filtering: 100%
 5354/5354 [00:01<00:00, 3745.02it/s]

✅ Kept 5354/5354 (100.0% quality rate)
\n🔍 Removing duplicates...
Encoding 5354 texts for duplicate detection...

Batches: 100%
 168/168 [00:10<00:00, 42.76it/s]
Duplicate removal: 100%
 5353/5353 [00:02<00:00, 1359.39it/s]

✅ Kept 2581/5354 unique samples (removed 2773 duplicates)
\n📝 Mapping titles to augmented samples...
✅ Assigned 39 unique titles to augmented samples
\n======================================================================
✅ NEUTRAL CLASS COMPLETE!
======================================================================
📊 Original: 2677
📊 Augmented: 2581
📊 Total: 5258
📊 Multiplier: 1.96x
\n======================================================================
✅ AUGMENTATION COMPLETE!
======================================================================


======================================================================
💾 COMBINING AND SAVING DATASET
======================================================================
\n✅ Saved to: augmented_adjudications_2025-10-22.csv
\n📊 Final Dataset Statistics:
   • Total samples: 13131
   • Original samples: 9965
   • Augmented samples: 3166
   • Augmentation rate: 31.8%
\n📊 Final Sentiment Distribution:
Final Sentiment
negative    5905
neutral     5843
positive    1383
Name: count, dtype: int64
\n📊 Final Polarization Distribution:
Final Polarization
partisan         7548
non_polarized    4103
objective        1480
Name: count, dtype: int64
\n🎯 Class Improvements:
   • Objective: 588 → 1480 (+151.7%)
   • Neutral: 2677 → 5843 (+118.3%)
\n======================================================================
📥 DOWNLOADING AUGMENTED DATASET...
======================================================================

\n✅ Downloaded: augmented_adjudications_2025-10-22.csv
\n======================================================================
🎉 AUGMENTATION COMPLETE!
======================================================================
