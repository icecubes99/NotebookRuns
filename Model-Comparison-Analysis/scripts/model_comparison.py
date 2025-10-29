"""
Model Performance Comparison Tool
Analyzes and visualizes performance metrics across MBERT, XLM-R, and RemBERT models
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def parse_run_data():
    """Parse run data from markdown files"""
    
    models = {
        'mBERT': {
            'path': Path('../[M]-MBERT/run-data.md'),
            'test_metrics': {
                'sent_acc': 0.901530,
                'sent_prec': 0.903125,
                'sent_rec': 0.905780,
                'sent_f1': 0.904442,
                'pol_acc': 0.868367,
                'pol_prec': 0.865420,
                'pol_rec': 0.872150,
                'pol_f1': 0.868765,
                'macro_f1_avg': 0.886604,
                'runtime': 4.2996,
                'training_time': '1.9h 52m',
                'total_time': '1.9h 55m'
            }
        },
        'XLM-R': {
            'path': Path('../[M]-XLMR/run-data.md'),
            'test_metrics': {
                'sent_acc': 0.915816,
                'sent_prec': 0.918235,
                'sent_rec': 0.920156,
                'sent_f1': 0.919143,
                'pol_acc': 0.885204,
                'pol_prec': 0.882145,
                'pol_rec': 0.887850,
                'pol_f1': 0.884920,
                'macro_f1_avg': 0.902032,
                'runtime': 4.423,
                'training_time': '2.5h 30m',
                'total_time': '5.4h 27m'
            }
        },
        'RemBERT': {
            'path': Path('../[M]-REMBERT/run-data.md'),
            'test_metrics': {
                'sent_acc': 0.908163,
                'sent_prec': 0.910580,
                'sent_rec': 0.912345,
                'sent_f1': 0.911453,
                'pol_acc': 0.875510,
                'pol_prec': 0.873285,
                'pol_rec': 0.878920,
                'pol_f1': 0.876085,
                'macro_f1_avg': 0.893769,
                'runtime': 16.8042,
                'training_time': '9.5h 31m',
                'total_time': '10.6h 35m'
            }
        }
    }
    
    return models

def create_comparison_dataframe(models):
    """Create a DataFrame for easy comparison"""
    
    data = []
    for model_name, model_data in models.items():
        metrics = model_data['test_metrics']
        row = {'Model': model_name, **metrics}
        data.append(row)
    
    df = pd.DataFrame(data)
    return df

def plot_sentiment_metrics(df):
    """Plot sentiment analysis metrics comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Sentiment Analysis Metrics Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['sent_acc', 'sent_prec', 'sent_rec', 'sent_f1']
    titles = ['Sentiment Accuracy', 'Sentiment Precision', 'Sentiment Recall', 'Sentiment F1']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(df['Model'], df[metric], color=['#3498db', '#e74c3c', '#2ecc71'])
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_ylim([0.94, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('../visualizations/sentiment_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/sentiment_metrics_comparison.png")
    plt.close()

def plot_polarization_metrics(df):
    """Plot polarization detection metrics comparison"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Polarization Detection Metrics Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['pol_acc', 'pol_prec', 'pol_rec', 'pol_f1', 'macro_f1_avg']
    titles = ['Polarization Accuracy', 'Polarization Precision', 'Polarization Recall', 
              'Polarization F1', 'Macro F1 Average']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3, idx % 3]
        bars = ax.bar(df['Model'], df[metric], color=colors)
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_ylim([0.80, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10)
    
    # Hide the last subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('../visualizations/polarization_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/polarization_metrics_comparison.png")
    plt.close()

def plot_overall_comparison(df):
    """Plot overall performance comparison radar chart"""
    
    # Select key metrics for radar chart
    categories = ['Sentiment F1', 'Polarization F1', 'Sent Accuracy', 
                  'Pol Accuracy', 'Macro F1']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Number of variables
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0.80, 1.0)
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, row in df.iterrows():
        values = [row['sent_f1'], row['pol_f1'], row['sent_acc'], 
                 row['pol_acc'], row['macro_f1_avg']]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], 
               color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.set_title('Overall Model Performance Comparison', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('../visualizations/overall_performance_radar.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/overall_performance_radar.png")
    plt.close()

def plot_efficiency_comparison(df):
    """Plot training time vs performance"""
    
    # Convert training time to minutes
    time_map = {
        'mBERT': 1.9 * 60 + 52,
        'XLM-R': 2.5 * 60 + 30,
        'RemBERT': 9.5 * 60 + 31
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, row in df.iterrows():
        model = row['Model']
        time_mins = time_map[model]
        ax.scatter(time_mins, row['macro_f1_avg'], s=500, 
                  label=model, alpha=0.6)
        ax.annotate(model, (time_mins, row['macro_f1_avg']),
                   fontsize=12, fontweight='bold',
                   xytext=(10, 10), textcoords='offset points')
    
    ax.set_xlabel('Training Time (minutes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Macro F1 Average', fontsize=12, fontweight='bold')
    ax.set_title('Training Efficiency: Performance vs Time', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('../visualizations/efficiency_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/efficiency_comparison.png")
    plt.close()

def create_metrics_heatmap(df):
    """Create a heatmap of all metrics"""
    
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    metrics_df = df[['Model'] + list(numeric_cols)]
    metrics_df = metrics_df.set_index('Model')
    
    # Remove runtime columns for better visualization
    metrics_df = metrics_df.drop(['runtime'], axis=1, errors='ignore')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.heatmap(metrics_df.T, annot=True, fmt='.4f', cmap='RdYlGn', 
               center=0.9, vmin=0.80, vmax=1.0, ax=ax,
               cbar_kws={'label': 'Score'})
    
    ax.set_title('Performance Metrics Heatmap', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metrics', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../visualizations/metrics_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/metrics_heatmap.png")
    plt.close()

def generate_summary_table(df):
    """Generate a formatted summary table"""
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Save to CSV
    df.to_csv('../results/model_comparison_summary.csv', index=False)
    print("✓ Saved: results/model_comparison_summary.csv")
    
    # Print winner for each metric
    print("\n" + "="*80)
    print("BEST PERFORMING MODEL PER METRIC")
    print("="*80)
    
    metrics_to_check = ['sent_acc', 'sent_f1', 'pol_acc', 'pol_f1', 'macro_f1_avg']
    for metric in metrics_to_check:
        best_idx = df[metric].idxmax()
        best_model = df.loc[best_idx, 'Model']
        best_score = df.loc[best_idx, metric]
        print(f"{metric:20s}: {best_model:10s} ({best_score:.4f})")
    print("="*80)

def main():
    """Main execution function"""
    
    print("🚀 Model Comparison Analysis Tool")
    print("="*80)
    
    # Parse data
    print("\n📊 Parsing run data...")
    models = parse_run_data()
    df = create_comparison_dataframe(models)
    
    # Generate summary table
    generate_summary_table(df)
    
    # Create visualizations
    print("\n📈 Generating visualizations...")
    print("-" * 80)
    
    plot_sentiment_metrics(df)
    plot_polarization_metrics(df)
    plot_overall_comparison(df)
    plot_efficiency_comparison(df)
    create_metrics_heatmap(df)
    
    print("\n✅ Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()
