# 📊 Model Comparison Analysis

**Location:** `Model-Comparison-Analysis/`

A complete toolkit for analyzing and comparing the performance of mBERT, XLM-RoBERTa, and RemBERT models across sentiment analysis and polarization detection tasks.

## ✅ What's Inside

### 📄 Documentation
- **README.md** - Complete project documentation
- **QUICK_REFERENCE.md** - Quick command reference guide
- **requirements.txt** - Python package dependencies

### 🐍 Scripts
- **model_comparison.py** - Full analysis with all visualizations
- **quick_stats.py** - Fast command-line statistics viewer
- **model_comparison_analysis.ipynb** - Interactive Jupyter notebook

### 📊 Generated Results
- **model_comparison_summary.csv** - Complete metrics table
- **5 PNG visualizations** - High-resolution comparison charts

## 🚀 Getting Started

```powershell
# 1. Activate virtual environment (from root)
.\venv\Scripts\Activate.ps1

# 2. Navigate to analysis folder
cd Model-Comparison-Analysis

# 3. View quick stats
python quick_stats.py

# 4. Generate full analysis
python model_comparison.py

# 5. Interactive analysis
jupyter notebook model_comparison_analysis.ipynb
```

## 📈 Key Results Summary

| Model   | Macro F1 | Training Time | Best For |
|---------|----------|---------------|----------|
| XLM-R   | 91.89%   | 2.5 hours     | Best overall performance ⭐ |
| RemBERT | 91.72%   | 9.5 hours     | Middle-tier option |
| mBERT   | 91.58%   | 1.9 hours     | Fast iteration & sentiment |

## 📦 All Files Organized

All comparison analysis files are now organized in the `Model-Comparison-Analysis/` folder:
- ✅ Python scripts
- ✅ Jupyter notebook
- ✅ Documentation files
- ✅ Generated visualizations
- ✅ CSV results
- ✅ Package requirements

The scripts automatically read model data from the parent directories:
- `../[M]-MBERT/run-data.md`
- `../[M]-XLMR/run-data.md`
- `../[M]-REMBERT/run-data.md`

## 📖 Learn More

See the full README in the `Model-Comparison-Analysis/` folder for:
- Detailed usage instructions
- Visualization descriptions
- Customization options
- Package information

---

**Created:** October 29, 2025  
**Purpose:** Organized comparison analysis toolkit for model evaluation
