# Model Performance Comparison Analysis

This project provides tools to compare the performance of three multilingual transformer models: **mBERT**, **XLM-RoBERTa**, and **RemBERT** for sentiment analysis and polarization detection tasks.

## Virtual Environment Setup

A Python virtual environment has been created with all necessary dependencies.

### Activating the Virtual Environment

**Note:** The virtual environment is located in the parent directory (`../venv/`)

**PowerShell:**
```powershell
cd ..
.\venv\Scripts\Activate.ps1
cd Model-Comparison-Analysis
```

**Command Prompt:**
```cmd
cd ..
.\venv\Scripts\activate.bat
cd Model-Comparison-Analysis
```

### Installed Packages

- pandas (2.3.3) - Data manipulation and analysis
- matplotlib (3.10.7) - Plotting and visualization
- seaborn (0.13.2) - Statistical data visualization
- numpy (2.3.4) - Numerical computing
- scikit-learn (1.7.2) - Machine learning utilities
- jupyter (1.1.1) - Notebook interface

## Usage

### Running the Comparison Analysis

```powershell
python model_comparison.py
```

This script will:
1. Parse run data from the three model directories
2. Generate a comprehensive comparison table
3. Create multiple visualization plots
4. Save all results to files

### Generated Outputs

#### CSV File
- `model_comparison_summary.csv` - Complete metrics table

#### Visualization Plots (PNG)
- `sentiment_metrics_comparison.png` - Sentiment analysis metrics (Accuracy, Precision, Recall, F1)
- `polarization_metrics_comparison.png` - Polarization detection metrics
- `overall_performance_radar.png` - Radar chart comparing all key metrics
- `efficiency_comparison.png` - Training time vs performance scatter plot
- `metrics_heatmap.png` - Heatmap of all metrics across models

## Model Performance Summary

| Model   | Sent F1  | Pol F1   | Macro F1 | Training Time |
|---------|----------|----------|----------|---------------|
| mBERT   | 0.9851   | 0.8464   | 0.9158   | 1.9h 52m      |
| XLM-R   | 0.9718   | 0.8661   | 0.9189   | 2.5h 30m      |
| RemBERT | 0.9791   | 0.8573   | 0.9172   | 9.5h 31m      |

### Key Findings

**Best Overall Performance:** XLM-R
- Highest macro F1 average: **91.89%**
- Best polarization F1: **86.61%**
- Moderate training time: **2.5 hours**

**Best Sentiment Analysis:** mBERT
- Highest sentiment F1: **98.51%**
- Fastest training: **1.9 hours**
- Strong overall performance: 91.58% macro F1

**Middle Performance:** RemBERT
- Between mBERT and XLM-R in most metrics
- Good sentiment F1: **97.91%**
- Trade-off: Longest training time (**9.5 hours**) with middle-tier performance

## Project Structure

```
NotebookRuns/
├── venv/                                    # Virtual environment (root level)
├── Model-Comparison-Analysis/              # This folder
│   ├── model_comparison.py                 # Main analysis script
│   ├── quick_stats.py                      # Quick stats viewer
│   ├── model_comparison_analysis.ipynb     # Interactive notebook
│   ├── requirements.txt                    # Package dependencies
│   ├── README.md                           # This file
│   ├── model_comparison_summary.csv        # Generated results
│   ├── sentiment_metrics_comparison.png    # Sentiment visualizations
│   ├── polarization_metrics_comparison.png # Polarization visualizations
│   ├── overall_performance_radar.png       # Radar chart
│   ├── efficiency_comparison.png           # Training time analysis
│   └── metrics_heatmap.png                 # Metrics heatmap
├── [M]-MBERT/
│   └── run-data.md                         # mBERT training results
├── [M]-XLMR/
│   └── run-data.md                         # XLM-R training results
└── [M]-REMBERT/
    └── run-data.md                         # RemBERT training results
```

## Customization

To add or modify metrics in the comparison:

1. Edit the `parse_run_data()` function in `model_comparison.py`
2. Update the visualization functions to include new metrics
3. Run the script again to regenerate outputs

## Requirements

- Python 3.13+
- See `requirements.txt` for package versions

## License

This analysis tool is for research and educational purposes.
