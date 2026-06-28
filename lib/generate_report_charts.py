import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'model_comparison')
os.makedirs(OUTPUT_DIR, exist_ok=True)
SUMMARY_PATH = os.path.join(OUTPUT_DIR, 'evaluation_summary.json')

def plot_charts(results_dict, suffix):
    sns.set_theme(style='whitegrid')
    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    error_rates = []
    for (model_name, metrics) in results_dict.items():
        model_names.append(model_name.replace('_', '\n'))
        accuracies.append(metrics['accuracy'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1_scores.append(metrics['f1_score'])
        error_rates.append(metrics.get('error_rate', 0) * 100)
    (fig, axes) = plt.subplots(1, 2, figsize=(18, 7))
    x = np.arange(len(model_names))
    width = 0.2
    ax1 = axes[0]
    bars1 = ax1.bar(x - 1.5 * width, accuracies, width, label='Accuracy', color='#3498db', edgecolor='white')
    bars2 = ax1.bar(x - 0.5 * width, precisions, width, label='Precision', color='#2ecc71', edgecolor='white')
    bars3 = ax1.bar(x + 0.5 * width, recalls, width, label='Recall', color='#e67e22', edgecolor='white')
    bars4 = ax1.bar(x + 1.5 * width, f1_scores, width, label='F1-Score', color='#e74c3c', edgecolor='white')
    ax1.set_xlabel('Mô hình', fontsize=12)
    ax1.set_ylabel('Giá trị', fontsize=12)
    ax1.set_title(f'So sánh Metrics các mô hình {suffix}', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=9)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=7)
    ax2 = axes[1]
    colors = ['#e74c3c' if er == max(error_rates) else '#3498db' for er in error_rates]
    bars = ax2.bar(model_names, error_rates, color=colors, edgecolor='white', width=0.6)
    ax2.set_xlabel('Mô hình', fontsize=12)
    ax2.set_ylabel('Tỷ lệ lỗi (%)', fontsize=12)
    ax2.set_title(f'Tỷ lệ dự đoán sai {suffix}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for (bar, er) in zip(bars, error_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.2, f'{er:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    plt.tight_layout()
    suffix_clean = suffix.replace(' ', '_').replace('(', '').replace(')', '').lower().strip('_')
    filename = f'model_comparison_{suffix_clean}.png'
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    if not os.path.exists(SUMMARY_PATH):
        return
    with open(SUMMARY_PATH, 'r', encoding='utf-8') as f:
        summary_json = json.load(f)
    holdout_results = {}
    external_results = {}
    for (model_name, data) in summary_json.items():
        if 'holdout' in data:
            holdout_results[model_name] = data['holdout']
        elif 'accuracy' in data and 'holdout' not in summary_json[next(iter(summary_json))]:
            holdout_results[model_name] = data
        if 'external' in data:
            external_results[model_name] = data['external']
    if holdout_results:
        plot_charts(holdout_results, '(Hold-out)')
    if external_results:
        plot_charts(external_results, '(External)')
if __name__ == '__main__':
    main()
