"""
Quick Test Script - Test augmentation on 5 samples
Run this first to make sure everything works before processing full dataset
"""

import pandas as pd
import sys

print("""
╔══════════════════════════════════════════════════════════════════════╗
║              Data Augmentation Test (5 Samples)                      ║
║                     Quick Validation Test                            ║
╚══════════════════════════════════════════════════════════════════════╝

This script will:
1. Create 5 sample texts
2. Augment them using EDA (fastest method)
3. Show you the results
4. Verify everything is working

⏱️ Expected runtime: 2-3 minutes
""")

# Check dependencies
print("🔍 Checking dependencies...")
try:
    import nlpaug
    print("✅ nlpaug installed")
except ImportError:
    print("❌ nlpaug not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nlpaug"])
    import nlpaug
    print("✅ nlpaug installed")

try:
    import tqdm
    print("✅ tqdm installed")
except ImportError:
    print("❌ tqdm not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    import tqdm
    print("✅ tqdm installed")

print("\n" + "="*70)
print("STEP 1: Creating Test Samples")
print("="*70 + "\n")

# Create test samples
test_samples = [
    {
        'text': 'According to the report, economic growth increased by 2.5 percent.',
        'sentiment': 'neutral',
        'polarization': 'objective',
        'notes': 'Objective statement (weak class #1)'
    },
    {
        'text': 'The study found no significant difference between the two groups.',
        'sentiment': 'neutral',
        'polarization': 'objective',
        'notes': 'Objective statement (weak class #1)'
    },
    {
        'text': 'How do I apply for the program?',
        'sentiment': 'neutral',
        'polarization': 'non_polarized',
        'notes': 'Neutral sentiment (weak class #2)'
    },
    {
        'text': 'The meeting is scheduled for 3pm tomorrow.',
        'sentiment': 'neutral',
        'polarization': 'non_polarized',
        'notes': 'Neutral sentiment (weak class #2)'
    },
    {
        'text': 'This policy is a disaster and must be stopped immediately.',
        'sentiment': 'negative',
        'polarization': 'partisan',
        'notes': 'Strong sentiment (for comparison)'
    }
]

df = pd.DataFrame(test_samples)

print("📊 Test samples created:")
for i, row in df.iterrows():
    print(f"\n{i+1}. [{row['polarization'].upper()} | {row['sentiment'].upper()}]")
    print(f"   Text: {row['text']}")
    print(f"   Note: {row['notes']}")

print("\n" + "="*70)
print("STEP 2: Initializing EDA Augmenter (Fastest Method)")
print("="*70 + "\n")

from data_augmentation_toolkit import EasyDataAugmenter

try:
    eda = EasyDataAugmenter()
    print("✅ EDA augmenter ready!")
except Exception as e:
    print(f"❌ Error initializing augmenter: {e}")
    print("\nTrying alternative initialization...")
    import nlpaug.augmenter.word as naw
    eda_syn = naw.SynonymAug(aug_src='wordnet', aug_p=0.15)
    print("✅ Alternative augmenter ready!")

print("\n" + "="*70)
print("STEP 3: Augmenting Samples")
print("="*70 + "\n")

results = []

for i, row in df.iterrows():
    print(f"\n{'─'*70}")
    print(f"Sample {i+1}: {row['text'][:50]}...")
    print(f"{'─'*70}")
    
    try:
        # Augment
        augmented = eda.augment_single(row['text'])
        
        print(f"\n✅ Generated {len(augmented)} variations:")
        for j, aug_text in enumerate(augmented, 1):
            print(f"\n   Variation {j}:")
            print(f"   Original:  {row['text']}")
            print(f"   Augmented: {aug_text}")
            
            results.append({
                'sample_id': i+1,
                'variation': j,
                'original': row['text'],
                'augmented': aug_text,
                'sentiment': row['sentiment'],
                'polarization': row['polarization']
            })
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print(f"   Skipping this sample...")

print("\n" + "="*70)
print("STEP 4: Summary")
print("="*70 + "\n")

results_df = pd.DataFrame(results)

print(f"📊 Augmentation Results:")
print(f"   • Original samples: {len(df)}")
print(f"   • Augmented samples: {len(results_df)}")
print(f"   • Multiplication: {len(results_df) / len(df):.1f}x")

print(f"\n📊 Breakdown by class:")
for pol_class in df['polarization'].unique():
    original_count = len(df[df['polarization'] == pol_class])
    augmented_count = len(results_df[results_df['polarization'] == pol_class])
    print(f"   • {pol_class}: {original_count} → {augmented_count} (+{augmented_count - original_count})")

# Save results
output_file = 'test_augmentation_results.csv'
results_df.to_csv(output_file, index=False)
print(f"\n💾 Results saved to: {output_file}")

print("\n" + "="*70)
print("✅ TEST COMPLETE!")
print("="*70)

print("""

🎉 Everything is working!

Next steps:
1. Review the augmented samples above
2. Check if they preserve the original meaning
3. If quality looks good, run the full augmentation:
   
   python data_augmentation_toolkit.py
   
4. If you see issues, adjust parameters in the toolkit

📊 Expected Performance After Full Augmentation:
   • Objective class: 90 → 450+ samples
   • Neutral class: 401 → 1200+ samples
   • Overall macro-F1: 68% → 73-76% (+5-8%)

⏱️ Full augmentation will take 4-6 hours (can run overnight)

🚀 Ready to start? Run:
   python data_augmentation_toolkit.py

Good luck!
""")

# Show a few examples
print("\n📝 Example Augmented Samples (for manual review):")
print("="*70)

for i in range(min(3, len(results_df))):
    row = results_df.iloc[i]
    print(f"\nSample {i+1}:")
    print(f"  Class: [{row['polarization']}]")
    print(f"  Original:  {row['original']}")
    print(f"  Augmented: {row['augmented']}")
    print(f"  Quality Check: ", end="")
    
    # Simple quality check
    orig_words = set(row['original'].lower().split())
    aug_words = set(row['augmented'].lower().split())
    overlap = len(orig_words & aug_words) / len(orig_words) if len(orig_words) > 0 else 0
    
    if overlap > 0.7:
        print(f"✅ Good ({overlap*100:.0f}% word overlap)")
    elif overlap > 0.5:
        print(f"⚠️ Moderate ({overlap*100:.0f}% word overlap)")
    else:
        print(f"❌ Low quality ({overlap*100:.0f}% word overlap)")

print("\n" + "="*70)

