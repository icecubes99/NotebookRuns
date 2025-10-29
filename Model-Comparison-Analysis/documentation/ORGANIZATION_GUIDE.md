# Model Comparison Analysis - Organized Structure

## 📁 Folder Organization

```
Model-Comparison-Analysis/
│
├── 📄 documentation/          # All documentation files
│   ├── README.md             # Complete project documentation
│   └── QUICK_REFERENCE.md    # Quick command reference
│
├── 🐍 scripts/               # All Python scripts and notebooks
│   ├── model_comparison.py           # Full analysis with all visualizations
│   ├── quick_stats.py                # Quick stats CLI tool
│   ├── custom_visualizations.py      # Custom accuracy & heatmap charts
│   ├── model_comparison_analysis.ipynb # Interactive Jupyter notebook
│   └── requirements.txt              # Package dependencies
│
├── 📊 visualizations/         # All generated charts and graphs
│   ├── ⭐ sentiment_accuracy_comparison.png     # NEW: Sentiment accuracy only
│   ├── ⭐ polarization_accuracy_comparison.png  # NEW: Polarization accuracy only
│   ├── ⭐ sentiment_metrics_heatmap.png         # NEW: Sentiment detailed heatmap
│   ├── ⭐ polarization_metrics_heatmap.png      # NEW: Polarization detailed heatmap
│   ├── sentiment_metrics_comparison.png         # Full sentiment metrics
│   ├── polarization_metrics_comparison.png      # Full polarization metrics
│   ├── overall_performance_radar.png            # Radar chart
│   ├── efficiency_comparison.png                # Time vs performance
│   └── metrics_heatmap.png                      # All metrics heatmap
│
└── 📈 results/                # Generated data files
    └── model_comparison_summary.csv  # Complete metrics table
```

## 🚀 Quick Start

### From Model-Comparison-Analysis root:

```powershell
# Activate venv (from parent directory)
cd ..
.\venv\Scripts\Activate.ps1
cd Model-Comparison-Analysis
```

### Run Scripts:

```powershell
# Navigate to scripts folder
cd scripts

# Quick stats view
python quick_stats.py

# Full analysis
python model_comparison.py

# Custom visualizations (accuracy + detailed heatmaps)
python custom_visualizations.py

# Interactive notebook
jupyter notebook model_comparison_analysis.ipynb
```

## ⭐ New Custom Visualizations

### 1. Sentiment Accuracy Comparison (`sentiment_accuracy_comparison.png`)
- **Standalone bar chart** for sentiment analysis accuracy only
- Blue color scheme (dark to light gradient)
- Shows percentage values and highlights winner with gold border
- Winner badge with "⭐ Best" label

### 2. Polarization Accuracy Comparison (`polarization_accuracy_comparison.png`)
- **Standalone bar chart** for polarization detection accuracy only
- Blue color scheme (dark to light gradient)
- Shows percentage values and highlights winner with gold border
- Winner badge with "⭐ Best" label

### 3. Sentiment Metrics Heatmap (`sentiment_metrics_heatmap.png`)
- **Detailed heatmap** for sentiment analysis metrics only
- Metrics: Accuracy, Precision, Recall, F1-Score
- Blue gradient color scheme
- Includes both decimal and percentage values

### 4. Polarization Metrics Heatmap (`polarization_metrics_heatmap.png`)
- **Detailed heatmap** for polarization detection metrics only
- Metrics: Accuracy, Precision, Recall, F1-Score
- Blue gradient color scheme
- Includes both decimal and percentage values

## 📊 All Visualizations

| File | Description | Type |
|------|-------------|------|
| **sentiment_accuracy_comparison.png** | Sentiment accuracy only | Bar Chart (Blue) |
| **polarization_accuracy_comparison.png** | Polarization accuracy only | Bar Chart (Blue) |
| **sentiment_metrics_heatmap.png** | Detailed sentiment metrics heatmap | Heatmap (Blue) |
| **polarization_metrics_heatmap.png** | Detailed polarization metrics heatmap | Heatmap (Blue) |
| sentiment_metrics_comparison.png | Full sentiment metrics (4 panels) | Bar Charts |
| polarization_metrics_comparison.png | Full polarization metrics (5 panels) | Bar Charts |
| overall_performance_radar.png | Overall performance comparison | Radar Chart |
| efficiency_comparison.png | Training time vs performance | Scatter Plot |
| metrics_heatmap.png | All metrics combined | Heatmap |

## 📈 Key Results

### Accuracy Rankings

**Sentiment Analysis Accuracy:**
1. mBERT: 98.16%
2. RemBERT: 97.54%
3. XLM-R: 96.38%

**Polarization Detection Accuracy:**
1. XLM-R: 87.04%
2. RemBERT: 85.82%
3. mBERT: 84.59%

### Overall Performance
1. **XLM-R**: 91.89% Macro F1 (Best overall, balanced performance)
2. **RemBERT**: 91.72% Macro F1 (Middle-tier, poor efficiency)
3. **mBERT**: 91.58% Macro F1 (Fastest training, best sentiment)

## 🛠️ Maintenance

### Regenerate All Visualizations
```powershell
cd scripts
python model_comparison.py        # Original visualizations
python custom_visualizations.py   # Custom visualizations
```

### Update Documentation
Edit files in `documentation/` folder

### View Results
- Check `results/` for CSV data
- Check `visualizations/` for all images

## 📝 Notes

- All scripts automatically reference correct paths
- Visualizations are high-res (300 DPI) for presentations
- CSV files compatible with Excel/Google Sheets
- Run scripts from the `scripts/` directory

---
**Last Updated:** October 29, 2025  
**Organization:** Improved folder structure with categorized subfolders
