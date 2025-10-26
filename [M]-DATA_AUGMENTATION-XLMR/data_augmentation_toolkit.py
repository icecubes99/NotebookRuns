"""
XLM-RoBERTa Data Augmentation Toolkit
Fast implementation for boosting weak classes (Objective, Neutral)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import time
from tqdm import tqdm

# ============================================================================
# SECTION 1: BACK-TRANSLATION AUGMENTATION (HIGHEST ROI)
# ============================================================================

class BackTranslationAugmenter:
    """
    Augment data via back-translation through multiple languages.
    Best for multilingual models like XLM-RoBERTa.
    """
    
    def __init__(self, intermediate_langs=['es', 'fr', 'de', 'ja']):
        """
        Args:
            intermediate_langs: Languages to translate through
        """
        try:
            from googletrans import Translator
            self.translator = Translator()
            self.intermediate_langs = intermediate_langs
        except ImportError:
            print("⚠️ googletrans not installed. Install with: pip install googletrans==4.0.0-rc1")
            raise
    
    def augment_single(self, text: str, src_lang='en') -> List[str]:
        """
        Augment a single text by back-translating through multiple languages.
        
        Returns:
            List of augmented versions (one per intermediate language)
        """
        augmented = []
        
        for lang in self.intermediate_langs:
            try:
                # Translate to intermediate language
                intermediate = self.translator.translate(
                    text, 
                    src=src_lang, 
                    dest=lang
                ).text
                time.sleep(0.5)  # Rate limiting
                
                # Translate back to source
                back = self.translator.translate(
                    intermediate, 
                    src=lang, 
                    dest=src_lang
                ).text
                time.sleep(0.5)
                
                augmented.append(back)
                
            except Exception as e:
                print(f"❌ Error translating via {lang}: {e}")
                continue
        
        return augmented
    
    def augment_batch(self, texts: List[str], src_lang='en') -> List[str]:
        """
        Augment a batch of texts.
        
        Returns:
            List of all augmented texts (len = len(texts) * len(intermediate_langs))
        """
        all_augmented = []
        
        print(f"🔄 Back-translating {len(texts)} samples through {len(self.intermediate_langs)} languages...")
        for text in tqdm(texts):
            augmented = self.augment_single(text, src_lang)
            all_augmented.extend(augmented)
        
        print(f"✅ Generated {len(all_augmented)} augmented samples!")
        return all_augmented


# ============================================================================
# SECTION 2: PARAPHRASING AUGMENTATION (HIGH QUALITY)
# ============================================================================

class ParaphraseAugmenter:
    """
    Augment data using T5 paraphrasing model.
    High quality, preserves meaning.
    """
    
    def __init__(self, model_name='t5-base', num_paraphrases=3):
        """
        Args:
            model_name: T5 model to use ('t5-base', 't5-large')
            num_paraphrases: Number of paraphrases per input
        """
        try:
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            print(f"📦 Loading {model_name} for paraphrasing...")
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.num_paraphrases = num_paraphrases
            print("✅ Model loaded!")
        except ImportError:
            print("⚠️ transformers not installed. Install with: pip install transformers")
            raise
    
    def paraphrase_single(self, text: str) -> List[str]:
        """
        Generate paraphrases for a single text.
        
        Returns:
            List of paraphrases
        """
        input_text = f"paraphrase: {text}"
        inputs = self.tokenizer.encode(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        outputs = self.model.generate(
            inputs,
            max_length=512,
            num_return_sequences=self.num_paraphrases,
            num_beams=self.num_paraphrases,
            temperature=1.5,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        
        paraphrases = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return paraphrases
    
    def paraphrase_batch(self, texts: List[str]) -> List[str]:
        """
        Paraphrase a batch of texts.
        
        Returns:
            List of all paraphrases (len = len(texts) * num_paraphrases)
        """
        all_paraphrases = []
        
        print(f"🔄 Paraphrasing {len(texts)} samples (x{self.num_paraphrases} each)...")
        for text in tqdm(texts):
            paraphrases = self.paraphrase_single(text)
            all_paraphrases.extend(paraphrases)
        
        print(f"✅ Generated {len(all_paraphrases)} paraphrases!")
        return all_paraphrases


# ============================================================================
# SECTION 3: EDA (EASY DATA AUGMENTATION) - FASTEST
# ============================================================================

class EasyDataAugmenter:
    """
    Simple augmentation using random operations.
    Fast and effective for most cases.
    """
    
    def __init__(self):
        try:
            import nlpaug.augmenter.word as naw
            # Synonym replacement
            self.syn_aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.15)
            # Random swap
            self.swap_aug = naw.RandomWordAug(action='swap', aug_p=0.15)
            # Random delete
            self.delete_aug = naw.RandomWordAug(action='delete', aug_p=0.1)
            print("✅ EDA augmenters ready!")
        except ImportError:
            print("⚠️ nlpaug not installed. Install with: pip install nlpaug")
            raise
    
    def augment_single(self, text: str) -> List[str]:
        """
        Apply all EDA techniques to a single text.
        
        Returns:
            List of augmented versions (3 variations)
        """
        augmented = []
        
        try:
            # Synonym replacement
            augmented.append(self.syn_aug.augment(text))
            # Random swap
            augmented.append(self.swap_aug.augment(text))
            # Random delete
            augmented.append(self.delete_aug.augment(text))
        except Exception as e:
            print(f"❌ EDA error: {e}")
        
        return augmented
    
    def augment_batch(self, texts: List[str]) -> List[str]:
        """
        Augment a batch of texts using EDA.
        
        Returns:
            List of all augmented texts (len = len(texts) * 3)
        """
        all_augmented = []
        
        print(f"🔄 Applying EDA to {len(texts)} samples...")
        for text in tqdm(texts):
            augmented = self.augment_single(text)
            all_augmented.extend(augmented)
        
        print(f"✅ Generated {len(all_augmented)} augmented samples!")
        return all_augmented


# ============================================================================
# SECTION 4: QUALITY FILTERING
# ============================================================================

class QualityFilter:
    """
    Filter augmented samples based on semantic similarity and diversity.
    """
    
    def __init__(self, similarity_threshold=0.75, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
        """
        Args:
            similarity_threshold: Minimum similarity to original (0.75 = keep if 75%+ similar)
            model_name: Sentence transformer model for computing similarity
        """
        try:
            from sentence_transformers import SentenceTransformer, util
            print(f"📦 Loading {model_name} for quality filtering...")
            self.model = SentenceTransformer(model_name)
            self.threshold = similarity_threshold
            self.util = util
            print("✅ Quality filter ready!")
        except ImportError:
            print("⚠️ sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
    
    def filter_augmented(
        self, 
        original_texts: List[str], 
        augmented_texts: List[str]
    ) -> Tuple[List[str], List[float]]:
        """
        Filter augmented texts based on semantic similarity to originals.
        
        Returns:
            Tuple of (filtered_augmented_texts, similarity_scores)
        """
        filtered = []
        scores = []
        
        print(f"🔍 Filtering {len(augmented_texts)} augmented samples...")
        
        # Compute embeddings
        orig_embeddings = self.model.encode(original_texts, convert_to_tensor=True)
        aug_embeddings = self.model.encode(augmented_texts, convert_to_tensor=True)
        
        # For each augmented text, find closest original
        for i, aug_emb in enumerate(tqdm(aug_embeddings)):
            # Compute similarity to all originals
            similarities = self.util.cos_sim(aug_emb, orig_embeddings)[0]
            max_similarity = similarities.max().item()
            
            # Keep if similar enough to some original
            if max_similarity >= self.threshold:
                filtered.append(augmented_texts[i])
                scores.append(max_similarity)
        
        print(f"✅ Kept {len(filtered)}/{len(augmented_texts)} samples (quality rate: {len(filtered)/len(augmented_texts)*100:.1f}%)")
        return filtered, scores
    
    def remove_duplicates(self, texts: List[str], similarity_threshold=0.95) -> List[str]:
        """
        Remove near-duplicate texts.
        
        Returns:
            List of unique texts
        """
        if len(texts) == 0:
            return texts
        
        print(f"🔍 Removing duplicates from {len(texts)} samples...")
        
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        unique_texts = [texts[0]]
        unique_embeddings = [embeddings[0]]
        
        for i in tqdm(range(1, len(texts))):
            # Check similarity to all unique texts so far
            similarities = self.util.cos_sim(embeddings[i], unique_embeddings)
            max_sim = similarities.max().item()
            
            # Keep if sufficiently different from all existing
            if max_sim < similarity_threshold:
                unique_texts.append(texts[i])
                unique_embeddings.append(embeddings[i])
        
        print(f"✅ Kept {len(unique_texts)}/{len(texts)} unique samples (removed {len(texts)-len(unique_texts)} duplicates)")
        return unique_texts


# ============================================================================
# SECTION 5: COMPLETE PIPELINE
# ============================================================================

class DataAugmentationPipeline:
    """
    Complete pipeline for data augmentation with quality control.
    """
    
    def __init__(
        self,
        use_backtranslation=True,
        use_paraphrasing=False,  # Slower but higher quality
        use_eda=True,
        quality_threshold=0.75
    ):
        self.use_backtranslation = use_backtranslation
        self.use_paraphrasing = use_paraphrasing
        self.use_eda = use_eda
        
        # Initialize augmenters
        if use_backtranslation:
            self.backtrans = BackTranslationAugmenter()
        if use_paraphrasing:
            self.paraphrase = ParaphraseAugmenter()
        if use_eda:
            self.eda = EasyDataAugmenter()
        
        # Initialize quality filter
        self.quality_filter = QualityFilter(similarity_threshold=quality_threshold)
    
    def augment_class(
        self, 
        df: pd.DataFrame, 
        class_column: str, 
        class_value: str, 
        text_column: str = 'text',
        target_multiplier: int = 5
    ) -> pd.DataFrame:
        """
        Augment all samples of a specific class.
        
        Args:
            df: Original dataframe
            class_column: Column name for class labels
            class_value: Value to augment (e.g., 'objective')
            text_column: Column name for text
            target_multiplier: Target multiplication factor (e.g., 5 = 5x more data)
        
        Returns:
            Dataframe with original + augmented samples
        """
        print(f"\n{'='*70}")
        print(f"🎯 AUGMENTING CLASS: {class_value}")
        print(f"{'='*70}\n")
        
        # Get samples for this class
        class_samples = df[df[class_column] == class_value]
        original_texts = class_samples[text_column].tolist()
        
        print(f"📊 Original samples: {len(original_texts)}")
        print(f"🎯 Target: {len(original_texts) * target_multiplier} samples (x{target_multiplier})")
        
        all_augmented = []
        
        # Apply augmentations
        if self.use_backtranslation:
            print(f"\n🔄 Step 1: Back-Translation")
            bt_augmented = self.backtrans.augment_batch(original_texts)
            all_augmented.extend(bt_augmented)
        
        if self.use_eda:
            print(f"\n🔄 Step 2: EDA (Easy Data Augmentation)")
            eda_augmented = self.eda.augment_batch(original_texts)
            all_augmented.extend(eda_augmented)
        
        if self.use_paraphrasing:
            print(f"\n🔄 Step 3: Paraphrasing (T5)")
            para_augmented = self.paraphrase.paraphrase_batch(original_texts)
            all_augmented.extend(para_augmented)
        
        print(f"\n📊 Total augmented (before filtering): {len(all_augmented)}")
        
        # Quality filtering
        print(f"\n🔍 Step 4: Quality Filtering")
        filtered_augmented, scores = self.quality_filter.filter_augmented(
            original_texts, 
            all_augmented
        )
        
        # Remove duplicates
        print(f"\n🔍 Step 5: Duplicate Removal")
        unique_augmented = self.quality_filter.remove_duplicates(filtered_augmented)
        
        # Limit to target multiplier
        target_count = len(original_texts) * (target_multiplier - 1)  # -1 because we keep originals
        if len(unique_augmented) > target_count:
            print(f"⚠️ Too many augmented samples ({len(unique_augmented)}), sampling {target_count}")
            unique_augmented = np.random.choice(unique_augmented, target_count, replace=False).tolist()
        
        # Create augmented dataframe
        aug_df = pd.DataFrame({
            text_column: unique_augmented,
            class_column: class_value,
            'is_augmented': True
        })
        
        # Add is_augmented column to original
        df_copy = df.copy()
        df_copy['is_augmented'] = False
        
        # Combine
        combined = pd.concat([df_copy, aug_df], ignore_index=True)
        
        print(f"\n{'='*70}")
        print(f"✅ AUGMENTATION COMPLETE FOR {class_value}")
        print(f"{'='*70}")
        print(f"📊 Original: {len(original_texts)}")
        print(f"📊 Augmented: {len(unique_augmented)}")
        print(f"📊 Total: {len(combined[combined[class_column] == class_value])}")
        print(f"📊 Multiplier: {len(combined[combined[class_column] == class_value]) / len(original_texts):.2f}x")
        print(f"{'='*70}\n")
        
        return combined


# ============================================================================
# SECTION 6: MAIN EXECUTION
# ============================================================================

def main():
    """
    Example usage: Augment objective and neutral classes.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                 XLM-RoBERTa Data Augmentation Toolkit                ║
    ║                     Fast Path to 75% Macro-F1                        ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Load your dataset
    print("📂 Loading dataset...")
    # df = pd.read_csv('your_data.csv')  # Replace with your data path
    
    # Example: Create sample data
    df = pd.DataFrame({
        'text': [
            'This is an objective statement from a study.',
            'Another neutral fact without opinion.',
            'I feel happy about this!',
            'This is terrible news.',
        ],
        'sentiment': ['neutral', 'neutral', 'positive', 'negative'],
        'polarization': ['objective', 'objective', 'partisan', 'partisan']
    })
    
    print(f"✅ Loaded {len(df)} samples")
    print(f"\nClass distribution:")
    print(df['polarization'].value_counts())
    print()
    
    # Initialize pipeline
    print("🔧 Initializing augmentation pipeline...")
    pipeline = DataAugmentationPipeline(
        use_backtranslation=True,   # Highest ROI
        use_paraphrasing=False,     # Slower, enable if you have time
        use_eda=True,               # Fast and effective
        quality_threshold=0.75
    )
    
    # Augment objective class (highest priority)
    print("\n" + "="*70)
    print("🎯 PHASE 1: AUGMENTING OBJECTIVE CLASS")
    print("="*70)
    df_augmented = pipeline.augment_class(
        df=df,
        class_column='polarization',
        class_value='objective',
        text_column='text',
        target_multiplier=5  # 5x more data
    )
    
    # Augment neutral class (second priority)
    print("\n" + "="*70)
    print("🎯 PHASE 2: AUGMENTING NEUTRAL CLASS")
    print("="*70)
    df_augmented = pipeline.augment_class(
        df=df_augmented,
        class_column='sentiment',
        class_value='neutral',
        text_column='text',
        target_multiplier=3  # 3x more data
    )
    
    # Save augmented dataset
    output_path = 'augmented_dataset.csv'
    df_augmented.to_csv(output_path, index=False)
    
    print(f"\n✅ AUGMENTATION COMPLETE!")
    print(f"💾 Saved to: {output_path}")
    print(f"\n📊 Final dataset size: {len(df_augmented)} samples")
    print(f"📊 Augmented samples: {df_augmented['is_augmented'].sum()}")
    print(f"📊 Original samples: {(~df_augmented['is_augmented']).sum()}")
    
    print(f"\n📊 Final class distribution:")
    print(df_augmented.groupby(['polarization', 'is_augmented']).size().unstack(fill_value=0))
    
    print("""
    
    🎉 Next Steps:
    1. Review augmented_dataset.csv
    2. Update your training notebook to use this dataset
    3. Adjust class weights (reduce objective/neutral weights)
    4. Reduce oversampling multipliers
    5. Train Run #12 and expect 73-76% macro-F1!
    
    Good luck! 🚀
    """)


if __name__ == "__main__":
    main()

