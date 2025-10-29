# NotebookRuns - Model Training & Comparison Project

Research project for training and comparing multilingual transformer models (mBERT, XLM-RoBERTa, RemBERT) for sentiment analysis and polarization detection tasks.

## 📁 Project Structure

```
NotebookRuns/
├── venv/                              # Python virtual environment
│
├── Model-Comparison-Analysis/         # 📊 Model comparison tools and results
│   ├── model_comparison.py           # Main analysis script
│   ├── quick_stats.py                # Quick stats viewer
│   ├── model_comparison_analysis.ipynb # Interactive notebook
│   ├── requirements.txt              # Package dependencies
│   ├── *.png                         # Generated visualizations
│   └── README.md                     # Detailed documentation
│
├── [M]-MBERT/                        # 🤖 mBERT model training
│   ├── MBERT_TRAINING.ipynb
│   ├── run-data.md
│   └── runs/                         # Training run history
│
├── [M]-XLMR/                         # 🤖 XLM-RoBERTa model training
│   ├── XLM_ROBERTA_TRAINING.ipynb
│   ├── run-data.md
│   └── runs/                         # Training run history
│
├── [M]-REMBERT/                      # 🤖 RemBERT model training
│   ├── REMBERT_TRAINING.ipynb
│   └── run-data.md
│
├── [M]-DATA_AUGMENTATION-XLMR/       # 📈 Data augmentation experiments
│   └── FILIPINO_DATA_AUGMENTATION_COLAB.ipynb
│
├── Runs/                             # 🗂️ Historical training runs
│   └── *.ipynb
│
├── archive/                          # 📦 Archived files
│   └── Various configuration and guide files
│
└── llm/                              # 🤖 LLM prompts and documentation
    └── cursor_*.md
```

## 🚀 Quick Start

### 1. Activate Virtual Environment

```powershell
# PowerShell
.\venv\Scripts\Activate.ps1
```

### 2. View Model Comparison

```powershell
cd Model-Comparison-Analysis
python quick_stats.py
```

### 3. Generate Full Analysis

```powershell
python model_comparison.py
```

### 4. Interactive Analysis

```powershell
jupyter notebook model_comparison_analysis.ipynb
```

## 📊 Model Performance Summary

| Model   | Sentiment F1 | Polarization F1 | Macro F1 | Training Time |
|---------|--------------|-----------------|----------|---------------|
| mBERT   | 98.51%       | 84.64%          | 91.58%   | 1.9h          |
| XLM-R   | 97.18%       | 86.61%          | 91.89%   | 2.5h          |
| RemBERT | 97.91%       | 85.73%          | 91.72%   | 9.5h          |

**Winner:** XLM-R achieves the highest overall performance (91.89%) with moderate training time, making it the most balanced choice.

## 🛠️ Technologies Used

- **Models:** mBERT, XLM-RoBERTa, RemBERT
- **Framework:** PyTorch, Transformers (HuggingFace)
- **Analysis:** Python, Pandas, Matplotlib, Seaborn
- **Environment:** Jupyter Notebooks, VS Code

## 📝 Key Files

- **Training Notebooks:** Located in `[M]-{MODEL}` directories
- **Run Data:** Each model folder contains `run-data.md` with detailed metrics
- **Comparison Tools:** All analysis scripts in `Model-Comparison-Analysis/`
- **Data:** CSV files for augmented adjudications and comments

## 🎯 Research Focus

This project focuses on:
1. **Sentiment Analysis** - Classifying text sentiment (negative, neutral, positive)
2. **Polarization Detection** - Identifying political polarization (non-polarized, objective, partisan)
3. **Multilingual Models** - Comparing performance across different transformer architectures
4. **Data Augmentation** - Enhancing training data for better model performance

## 📖 Documentation

- Full comparison analysis: See `Model-Comparison-Analysis/README.md`
- Individual model documentation: Check `model-mdver.md` files in model directories
- Training strategies: Review `archive/` for various implementation guides

## 🔗 Related Projects

- Model training runs: `[M]-MBERT`, `[M]-XLMR`, `[M]-REMBERT`
- Data augmentation: `[M]-DATA_AUGMENTATION-XLMR`
- Historical runs: `Runs/` directory

---

**Last Updated:** October 29, 2025
