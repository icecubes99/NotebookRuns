# Visualization Gallery

Quick reference guide for all generated visualizations.

**Color Scheme:** All custom visualizations use a professional blue color scheme.

## 🎯 Custom Visualizations (UPDATED)

### 1. Sentiment Accuracy Comparison
**File:** `visualizations/sentiment_accuracy_comparison.png`

Shows **sentiment analysis accuracy only**:
- Bar chart with three models
- Blue color scheme (dark to light blue)
- Gold border highlights the best performer (mBERT)
- Includes decimal values and percentages
- Winner badge with "⭐ Best" label

**Best for:** Quick sentiment accuracy comparison, presentations

---

### 2. Polarization Accuracy Comparison
**File:** `visualizations/polarization_accuracy_comparison.png`

Shows **polarization detection accuracy only**:
- Bar chart with three models
- Blue color scheme (dark to light blue)
- Gold border highlights the best performer (XLM-R)
- Includes decimal values and percentages
- Winner badge with "⭐ Best" label

**Best for:** Quick polarization accuracy comparison, presentations

---

### 3. Sentiment Metrics Heatmap
**File:** `visualizations/sentiment_metrics_heatmap.png`

Detailed heatmap showing **sentiment analysis only**:
- Rows: Accuracy, Precision, Recall, F1-Score
- Columns: mBERT, XLM-R, RemBERT
- Blue color gradient: Light (lower) → Dark (higher)
- Shows both decimal and percentage values

**Best for:** Detailed sentiment performance analysis, academic papers

---

### 4. Polarization Metrics Heatmap
**File:** `visualizations/polarization_metrics_heatmap.png`

Detailed heatmap showing **polarization detection only**:
- Rows: Accuracy, Precision, Recall, F1-Score
- Columns: mBERT, XLM-R, RemBERT
- Blue color gradient: Light (lower) → Dark (higher)
- Shows both decimal and percentage values

**Best for:** Detailed polarization performance analysis, academic papers

---

## 📊 Original Visualizations

### 4. Sentiment Metrics Comparison
**File:** `visualizations/sentiment_metrics_comparison.png`

4-panel bar chart showing:
- Sentiment Accuracy
- Sentiment Precision
- Sentiment Recall
- Sentiment F1-Score

**Best for:** Comprehensive sentiment analysis overview

---

### 5. Polarization Metrics Comparison
**File:** `visualizations/polarization_metrics_comparison.png`

5-panel bar chart showing:
- Polarization Accuracy
- Polarization Precision
- Polarization Recall
- Polarization F1-Score
- Macro F1 Average

**Best for:** Comprehensive polarization detection overview

---

### 6. Overall Performance Radar
**File:** `visualizations/overall_performance_radar.png`

Radar chart comparing:
- Sentiment F1
- Polarization F1
- Sentiment Accuracy
- Polarization Accuracy
- Macro F1

**Best for:** Overall model comparison, executive summaries

---

### 7. Efficiency Comparison
**File:** `visualizations/efficiency_comparison.png`

Scatter plot showing:
- X-axis: Training time (minutes)
- Y-axis: Macro F1 Average
- Each model plotted as a point

**Best for:** Training time vs performance trade-offs

---

### 8. Metrics Heatmap (Combined)
**File:** `visualizations/metrics_heatmap.png`

Combined heatmap showing:
- All sentiment metrics (acc, prec, rec, f1)
- All polarization metrics (acc, prec, rec, f1)
- Macro F1 Average

**Best for:** Complete overview, comparing all metrics at once

---

## 📋 Usage Recommendations

### For Presentations
- **Opening slide:** `overall_performance_radar.png`
- **Accuracy focus:** `accuracy_comparison.png`
- **Efficiency discussion:** `efficiency_comparison.png`

### For Academic Papers
- **Sentiment results:** `sentiment_metrics_heatmap.png`
- **Polarization results:** `polarization_metrics_heatmap.png`
- **Comprehensive comparison:** `metrics_heatmap.png`

### For Quick Reports
- **Accuracy only:** `accuracy_comparison.png`
- **Full metrics:** `sentiment_metrics_comparison.png` + `polarization_metrics_comparison.png`

---

## 🎨 Visualization Specifications

All images are:
- **Resolution:** 300 DPI (publication quality)
- **Format:** PNG with white background
- **Color Scheme:** Professional blue gradient
  - Dark Blue: #1f4788
  - Medium Blue: #2874A6
  - Light Blue: #5DADE2
- **Winner Highlight:** Gold border (#FFD700)
- **Heatmaps:** Blue gradient (light to dark)

---

**Generated:** October 29, 2025  
**Total Visualizations:** 9 (4 custom + 5 original)  
**Theme:** Professional Blue Color Scheme
