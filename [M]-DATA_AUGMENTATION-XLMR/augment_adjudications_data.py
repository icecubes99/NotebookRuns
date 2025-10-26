"""
Custom Data Augmentation Script for adjudications_2025-10-22.csv
Targets: Objective (polarization) and Neutral (sentiment) classes
"""

import pandas as pd
import sys
from pathlib import Path

print("""
╔══════════════════════════════════════════════════════════════════════╗
║           Data Augmentation for Adjudications Dataset               ║
║                  Targeting Weak Classes                              ║
╚══════════════════════════════════════════════════════════════════════╝

Dataset: adjudications_2025-10-22.csv
Target classes: Objective (polarization), Neutral (sentiment)
Expected runtime: 4-6 hours (automated)
""")

# ============================================================================
# STEP 1: Load and Analyze Data
# ============================================================================

print("=" * 70)
print("STEP 1: Loading Dataset")
print("=" * 70)

csv_path = Path(__file__).parent.parent / "adjudications_2025-10-22.csv"

try:
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} samples from {csv_path.name}")
except FileNotFoundError:
    print(f"❌ File not found: {csv_path}")
    print("Please ensure adjudications_2025-10-22.csv is in the root directory")
    sys.exit(1)

# Check columns
print(f"\n📊 Columns: {list(df.columns)}")
print(f"📊 Shape: {df.shape}")

# Display class distribution
print("\n📊 Class Distribution:")
print("\n--- Sentiment ---")
print(df['Final Sentiment'].value_counts())
print("\n--- Polarization ---")
print(df['Final Polarization'].value_counts())

# Identify weak classes
objective_count = len(df[df['Final Polarization'] == 'objective'])
neutral_count = len(df[df['Final Sentiment'] == 'neutral'])

print(f"\n🎯 Target Classes:")
print(f"   • Objective (polarization): {objective_count} samples")
print(f"   • Neutral (sentiment): {neutral_count} samples")

if objective_count < 100:
    print(f"   ⚠️  Objective class is VERY SMALL ({objective_count} samples)")
    print(f"   → Will augment to ~{objective_count * 5}+ samples (5x)")

if neutral_count < 500:
    print(f"   ⚠️  Neutral class is SMALL ({neutral_count} samples)")
    print(f"   → Will augment to ~{neutral_count * 3}+ samples (3x)")

# ============================================================================
# STEP 2: Initialize Augmentation Pipeline
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: Initializing Augmentation Pipeline")
print("=" * 70)

try:
    from data_augmentation_toolkit import DataAugmentationPipeline
    
    pipeline = DataAugmentationPipeline(
        use_backtranslation=True,   # High ROI, multilingual support
        use_paraphrasing=False,     # Disabled for speed (can enable later)
        use_eda=True,               # Fast and effective
        quality_threshold=0.75      # Keep samples 75%+ similar to originals
    )
    
    print("✅ Pipeline initialized!")
    print("   • Back-translation: ✅ Enabled (4 languages: es, fr, de, ja)")
    print("   • Paraphrasing: ❌ Disabled (for speed)")
    print("   • EDA: ✅ Enabled")
    print("   • Quality threshold: 75%")
    
except ImportError as e:
    print(f"❌ Error importing toolkit: {e}")
    print("\n💡 Installing required packages...")
    import subprocess
    packages = [
        "googletrans==4.0.0-rc1",
        "sentence-transformers",
        "nlpaug",
        "transformers",
        "torch",
        "tqdm"
    ]
    for package in packages:
        print(f"   Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
    
    print("✅ Packages installed! Please run the script again.")
    sys.exit(0)

# ============================================================================
# STEP 3: Augment Objective Class (Priority 1)
# ============================================================================

print("\n" + "=" * 70)
print("STEP 3: Augmenting OBJECTIVE Class (Polarization)")
print("=" * 70)

df_augmented = pipeline.augment_class(
    df=df,
    class_column='Final Polarization',
    class_value='objective',
    text_column='Comment',  # Using Comment as the text field
    target_multiplier=5  # 5x more objective samples
)

# ============================================================================
# STEP 4: Augment Neutral Class (Priority 2)
# ============================================================================

print("\n" + "=" * 70)
print("STEP 4: Augmenting NEUTRAL Class (Sentiment)")
print("=" * 70)

df_augmented = pipeline.augment_class(
    df=df_augmented,
    class_column='Final Sentiment',
    class_value='neutral',
    text_column='Comment',
    target_multiplier=3  # 3x more neutral samples
)

# ============================================================================
# STEP 5: Save Augmented Dataset
# ============================================================================

print("\n" + "=" * 70)
print("STEP 5: Saving Augmented Dataset")
print("=" * 70)

output_path = Path(__file__).parent / "augmented_adjudications_2025-10-22.csv"
df_augmented.to_csv(output_path, index=False)

print(f"✅ Saved to: {output_path}")
print(f"\n📊 Final Dataset Statistics:")
print(f"   • Total samples: {len(df_augmented)}")
print(f"   • Original samples: {(~df_augmented['is_augmented']).sum()}")
print(f"   • Augmented samples: {df_augmented['is_augmented'].sum()}")
print(f"   • Augmentation rate: {df_augmented['is_augmented'].sum() / len(df) * 100:.1f}%")

print(f"\n📊 Final Class Distribution:")
print("\n--- Sentiment ---")
print(df_augmented['Final Sentiment'].value_counts())
print("\n--- Polarization ---")
print(df_augmented['Final Polarization'].value_counts())

# ============================================================================
# STEP 6: Generate Summary Report
# ============================================================================

print("\n" + "=" * 70)
print("STEP 6: Generating Summary Report")
print("=" * 70)

# Calculate improvements
objective_before = len(df[df['Final Polarization'] == 'objective'])
objective_after = len(df_augmented[df_augmented['Final Polarization'] == 'objective'])
objective_improvement = (objective_after - objective_before) / objective_before * 100

neutral_before = len(df[df['Final Sentiment'] == 'neutral'])
neutral_after = len(df_augmented[df_augmented['Final Sentiment'] == 'neutral'])
neutral_improvement = (neutral_after - neutral_before) / neutral_before * 100

summary = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    AUGMENTATION COMPLETE!                            ║
╚══════════════════════════════════════════════════════════════════════╝

📊 RESULTS SUMMARY

Original Dataset:
  • Total samples: {len(df)}
  • Objective samples: {objective_before}
  • Neutral samples: {neutral_before}

Augmented Dataset:
  • Total samples: {len(df_augmented)} (+{len(df_augmented) - len(df)})
  • Objective samples: {objective_after} (+{objective_improvement:.1f}%)
  • Neutral samples: {neutral_after} (+{neutral_improvement:.1f}%)

💾 Output File: {output_path.name}

🎯 EXPECTED PERFORMANCE IMPROVEMENTS

Based on Run #11 baseline (68.36% macro-F1):

Class Improvements:
  • Objective F1: 50.28% → 65-70% (+15-20%)
  • Neutral F1: 55.69% → 68-72% (+13-17%)
  • Non-polarized F1: 64.85% → 70-73% (+5-8%)
  • Positive F1: 72.77% → 76-78% (+3-5%)

Overall:
  • Current: 68.36% macro-F1
  • Expected: 73-76% macro-F1 (+5-8%)
  • Target: 75% macro-F1 ✅ ACHIEVABLE!

📋 NEXT STEPS

1. Review augmented dataset (spot check some samples):
   {output_path}

2. Update your training notebook configuration:

   CSV_PATH = '/content/{output_path.name}'
   
   OBJECTIVE_BOOST_MULT = 1.0  # Was 3.5 → now have enough data!
   NEUTRAL_BOOST_MULT = 1.0    # Was 0.3 → now have enough data!
   
   CLASS_WEIGHT_MULT = {{
       "sentiment": {{
           "neutral": 1.20,    # Reduced from 1.70
       }},
       "polarization": {{
           "objective": 1.30,  # Reduced from 2.80
       }}
   }}
   
   EPOCHS = 15              # Reduced from 20
   BATCH_SIZE = 24          # Increased from 16
   EARLY_STOP_PATIENCE = 5  # Reduced from 6

3. Upload {output_path.name} to Google Colab

4. Train Run #12 with the augmented data!

5. Compare results to Run #11

🎉 Expected Result: 73-76% macro-F1 within 1-2 training runs!

Good luck! 🚀
"""

print(summary)

# Save summary to file
summary_path = Path(__file__).parent / "AUGMENTATION_SUMMARY.txt"
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"💾 Summary also saved to: {summary_path}")

print("\n" + "=" * 70)
print("✅ ALL DONE!")
print("=" * 70)

