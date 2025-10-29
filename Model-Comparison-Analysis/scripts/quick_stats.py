"""
Quick Stats - Simple command-line tool for viewing model comparison stats
"""

import pandas as pd

def main():
    print("\n" + "="*80)
    print("📊 MODEL PERFORMANCE QUICK STATS")
    print("="*80)
    
    # Load the CSV
    df = pd.read_csv('../results/model_comparison_summary.csv')
    
    print("\n🎯 OVERALL RANKINGS (by Macro F1 Average):")
    print("-" * 80)
    ranked = df[['Model', 'macro_f1_avg', 'training_time']].sort_values('macro_f1_avg', ascending=False)
    for idx, row in ranked.iterrows():
        rank = idx + 1
        print(f"  {rank}. {row['Model']:10s} - Macro F1: {row['macro_f1_avg']:.4f} (Training: {row['training_time']})")
    
    print("\n📈 SENTIMENT ANALYSIS BEST SCORES:")
    print("-" * 80)
    sent_metrics = ['sent_acc', 'sent_prec', 'sent_rec', 'sent_f1']
    for metric in sent_metrics:
        best_idx = df[metric].idxmax()
        model = df.loc[best_idx, 'Model']
        score = df.loc[best_idx, metric]
        print(f"  {metric:15s}: {model:10s} ({score:.4f})")
    
    print("\n🎭 POLARIZATION DETECTION BEST SCORES:")
    print("-" * 80)
    pol_metrics = ['pol_acc', 'pol_prec', 'pol_rec', 'pol_f1']
    for metric in pol_metrics:
        best_idx = df[metric].idxmax()
        model = df.loc[best_idx, 'Model']
        score = df.loc[best_idx, metric]
        print(f"  {metric:15s}: {model:10s} ({score:.4f})")
    
    print("\n⚡ EFFICIENCY COMPARISON:")
    print("-" * 80)
    for idx, row in df.iterrows():
        efficiency = row['macro_f1_avg'] / df.loc[idx, 'runtime']
        print(f"  {row['Model']:10s}: {row['macro_f1_avg']:.4f} F1 in {row['training_time']:10s} " +
              f"(Runtime: {row['runtime']:.2f}s, Eff: {efficiency:.4f})")
    
    print("\n💡 QUICK RECOMMENDATIONS:")
    print("-" * 80)
    best_performance = df.loc[df['macro_f1_avg'].idxmax(), 'Model']
    fastest = df.loc[df['runtime'].idxmin(), 'Model']
    
    print(f"  • Best Performance:    {best_performance} - Use when accuracy is critical")
    print(f"  • Fastest Training:    {fastest} - Use for rapid iteration/prototyping")
    print(f"  • Best Balance:        XLM-R - Good performance with reasonable training time")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
