#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Notebook Splitter
Creates two separate training notebooks from EDITABLE-FILE.ipynb
- MBERT-TRAINING.ipynb (trains only mBERT)
- XLM-ROBERTA-TRAINING.ipynb (trains only XLM-RoBERTa)
"""

import json
import sys
import io
from pathlib import Path

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def split_notebook():
    """Split the main notebook into two model-specific notebooks"""
    
    # Read the original notebook
    source_file = Path("EDITABLE-FILE.ipynb")
    
    if not source_file.exists():
        print(f"❌ Error: {source_file} not found!")
        print("📁 Make sure EDITABLE-FILE.ipynb is in the same directory as this script")
        return False
    
    print(f"📖 Reading {source_file}...")
    with open(source_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Create mBERT version
    print("\n🤖 Creating MBERT-TRAINING.ipynb...")
    mbert_notebook = json.loads(json.dumps(notebook))  # Deep copy
    
    # Find and modify the config cell (usually cell 8)
    for cell in mbert_notebook['cells']:
        if 'source' in cell and isinstance(cell['source'], list):
            source_text = ''.join(cell['source'])
            
            # Modify MODELS_TO_RUN
            if 'MODELS_TO_RUN = ["mbert", "xlm_roberta"]' in source_text:
                cell['source'] = [line.replace(
                    'MODELS_TO_RUN = ["mbert", "xlm_roberta"]',
                    'MODELS_TO_RUN = ["mbert"]  # ← TRAINING ONLY mBERT'
                ) for line in cell['source']]
            
            # Modify OUT_DIR
            if 'OUT_DIR = "./runs_multitask"' in source_text:
                cell['source'] = [line.replace(
                    'OUT_DIR = "./runs_multitask"',
                    'OUT_DIR = "./runs_mbert"  # ← Separate output directory'
                ) for line in cell['source']]
            
            # Add model identifier at top
            if '# ===== Section 3' in source_text and 'TRAINING ONLY mBERT' not in source_text:
                cell['source'].insert(0, '# 🤖 TRAINING ONLY: mBERT (bert-base-multilingual-cased)\\n')
                cell['source'].insert(1, '# Expected: ~35-40 min, 60-65% macro-F1\\n')
                cell['source'].insert(2, '\\n')
    
    # Save mBERT notebook
    mbert_file = Path("MBERT-TRAINING.ipynb")
    with open(mbert_file, 'w', encoding='utf-8') as f:
        json.dump(mbert_notebook, f, indent=1, ensure_ascii=False)
    print(f"✅ Created {mbert_file}")
    
    # Create XLM-RoBERTa version
    print("\n🤖 Creating XLM-ROBERTA-TRAINING.ipynb...")
    xlm_notebook = json.loads(json.dumps(notebook))  # Deep copy
    
    # Find and modify the config cell
    for cell in xlm_notebook['cells']:
        if 'source' in cell and isinstance(cell['source'], list):
            source_text = ''.join(cell['source'])
            
            # Modify MODELS_TO_RUN
            if 'MODELS_TO_RUN = ["mbert", "xlm_roberta"]' in source_text:
                cell['source'] = [line.replace(
                    'MODELS_TO_RUN = ["mbert", "xlm_roberta"]',
                    'MODELS_TO_RUN = ["xlm_roberta"]  # ← TRAINING ONLY XLM-RoBERTa'
                ) for line in cell['source']]
            
            # Modify OUT_DIR
            if 'OUT_DIR = "./runs_multitask"' in source_text:
                cell['source'] = [line.replace(
                    'OUT_DIR = "./runs_multitask"',
                    'OUT_DIR = "./runs_xlm_roberta"  # ← Separate output directory'
                ) for line in cell['source']]
            
            # Add model identifier at top
            if '# ===== Section 3' in source_text and 'TRAINING ONLY XLM' not in source_text:
                cell['source'].insert(0, '# 🤖 TRAINING ONLY: XLM-RoBERTa (xlm-roberta-base)\\n')
                cell['source'].insert(1, '# Expected: ~35-40 min, 65-70% macro-F1\\n')
                cell['source'].insert(2, '\\n')
    
    # Save XLM-RoBERTa notebook
    xlm_file = Path("XLM-ROBERTA-TRAINING.ipynb")
    with open(xlm_file, 'w', encoding='utf-8') as f:
        json.dump(xlm_notebook, f, indent=1, ensure_ascii=False)
    print(f"✅ Created {xlm_file}")
    
    print("\n" + "="*60)
    print("🎉 SUCCESS! Created two separate training notebooks:")
    print(f"   1. {mbert_file}")
    print(f"   2. {xlm_file}")
    print("\n📋 Next Steps:")
    print("   1. Upload both notebooks to separate Google Colab sessions")
    print("   2. Upload your CSV: adjudications_2025-10-22.csv")
    print("   3. Run both notebooks simultaneously!")
    print("   4. Combined training time: ~35-40 min (instead of 72 min)")
    print("="*60)
    
    return True

if __name__ == "__main__":
    print("🚀 Automated Notebook Splitter")
    print("="*60)
    
    success = split_notebook()
    
    if success:
        sys.exit(0)
    else:
        print("\n❌ Failed to create notebooks")
        sys.exit(1)

