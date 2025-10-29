# Model Comparison Analysis

**Clean, organized analysis toolkit for comparing mBERT, XLM-R, and RemBERT models**

## 📂 Folder Structure

```
Model-Comparison-Analysis/
├── 📄 documentation/     → All README files and guides
├── 🐍 scripts/           → Python scripts and notebooks
├── 📊 visualizations/    → All generated charts (8 images)
└── 📈 results/           → CSV data files
```

## 🚀 Quick Actions

```powershell
# View quick stats
cd scripts
python quick_stats.py

# Generate all visualizations
python model_comparison.py
python custom_visualizations.py

# Interactive analysis
jupyter notebook model_comparison_analysis.ipynb
```

## ⭐ New Custom Visualizations

Four new focused visualizations with **professional blue color scheme**:

1. **`sentiment_accuracy_comparison.png`** - Sentiment accuracy bar chart (separate)
2. **`polarization_accuracy_comparison.png`** - Polarization accuracy bar chart (separate)
3. **`sentiment_metrics_heatmap.png`** - Detailed sentiment metrics (blue heatmap)
4. **`polarization_metrics_heatmap.png`** - Detailed polarization metrics (blue heatmap)

## 📖 Documentation

- **[ORGANIZATION_GUIDE.md](documentation/ORGANIZATION_GUIDE.md)** - Complete folder structure guide
- **[VISUALIZATION_GUIDE.md](documentation/VISUALIZATION_GUIDE.md)** - Visual gallery and usage recommendations
- **[README.md](documentation/README.md)** - Full project documentation
- **[QUICK_REFERENCE.md](documentation/QUICK_REFERENCE.md)** - Command reference

## 📊 Results Summary

| Model   | Sentiment Acc | Polarization Acc | Macro F1 | Training Time |
|---------|---------------|------------------|----------|---------------|
| mBERT   | **98.16%** ⭐ | 84.59%          | 91.58%   | **1.9h** ⭐   |
| XLM-R   | 96.38%        | **87.04%** ⭐   | **91.89%** ⭐ | 2.5h   |
| RemBERT | 97.54%        | 85.82%          | 91.72%   | 9.5h          |

**Winner:** XLM-R (best overall performance with reasonable training time)

## 🎯 File Counts

- **Documentation:** 5 files (includes INDEX and QUICK_ACCESS)
- **Scripts:** 5 files
- **Visualizations:** 9 images (4 custom with blue theme!)
- **Results:** 1 CSV file

---

**Last Updated:** October 29, 2025  
**Status:** ✅ Fully organized with professional blue color scheme
