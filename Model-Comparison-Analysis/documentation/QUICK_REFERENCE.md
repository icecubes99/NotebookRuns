# Model Comparison Analysis - Quick Reference

## 📂 Location
```
D:\School\NotebookRuns\Model-Comparison-Analysis\
```

## 🚀 Quick Commands

### Activate Environment & Navigate
```powershell
# From NotebookRuns root directory
.\venv\Scripts\Activate.ps1
cd Model-Comparison-Analysis
```

### Run Analysis Tools
```powershell
# Quick stats summary (fast)
python quick_stats.py

# Full analysis with visualizations
python model_comparison.py

# Interactive notebook
jupyter notebook model_comparison_analysis.ipynb
```

## 📊 Generated Files

| File | Description |
|------|-------------|
| `model_comparison_summary.csv` | Complete metrics table in CSV format |
| `sentiment_metrics_comparison.png` | 4-panel sentiment analysis comparison |
| `polarization_metrics_comparison.png` | 5-panel polarization detection comparison |
| `overall_performance_radar.png` | Radar chart of all key metrics |
| `efficiency_comparison.png` | Training time vs performance scatter plot |
| `metrics_heatmap.png` | Heatmap of all metrics across models |

## 📈 Key Results

### Overall Rankings
1. **XLM-R** - 91.89% Macro F1 (2.5h training) ⭐
2. **RemBERT** - 91.72% Macro F1 (9.5h training)
3. **mBERT** - 91.58% Macro F1 (1.9h training)

### Best Use Cases
- **XLM-R**: Best overall performance with reasonable training time
- **mBERT**: Fast iteration, prototyping, best sentiment analysis
- **RemBERT**: Middle-tier option, not optimal for time vs performance

## 🔄 Regenerate Analysis

If you update the run-data.md files in the parent model directories:
```powershell
python model_comparison.py
```

This will regenerate all CSV and PNG files with updated data.

## 📝 Files in This Folder

```
Model-Comparison-Analysis/
├── README.md                           # Full documentation
├── QUICK_REFERENCE.md                  # This file
├── requirements.txt                    # Package dependencies
├── model_comparison.py                 # Main analysis script
├── quick_stats.py                      # Quick stats CLI tool
├── model_comparison_analysis.ipynb     # Interactive notebook
├── model_comparison_summary.csv        # Results table
└── *.png                               # 5 visualization files
```

## 💡 Tips

- Visualizations are high resolution (300 DPI) suitable for presentations
- CSV file can be imported into Excel or Google Sheets
- Notebook allows custom analysis and additional visualizations
- All scripts read from parent directories automatically

---
**Generated:** October 29, 2025
