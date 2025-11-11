"""
Visualization 4 of 8 for Section 10: Benchmark Performance Analysis

This script visualizes performance on standard benchmarks:
- MMLU subject breakdown
- HumanEval code generation performance
- TruthfulQA accuracy analysis
- Multi-task benchmark comparison

Output: visualization/images/evaluation/benchmark_performance.png
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook", font_scale=1.1)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

OUTPUT_DIR = "../../images/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("SECTION 10 - VISUALIZATION 4/8: Benchmark Performance Analysis")
print("=" * 80)

def create_benchmark_performance():
    """Four-panel benchmark analysis"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # Panel 1: MMLU Subject Breakdown
    ax1 = fig.add_subplot(gs[0, :])

    subjects = ['STEM', 'Humanities', 'Social\nSciences', 'Math', 'History',
                'Law', 'Medicine', 'Business', 'Philosophy', 'Psychology']
    base_scores = [42.5, 48.2, 51.8, 38.6, 52.3, 45.7, 41.2, 49.8, 50.1, 53.2]
    finetuned_scores = [58.9, 62.4, 65.1, 54.2, 68.5, 61.3, 72.8, 63.9, 64.2, 66.8]
    human_scores = [89.2, 92.1, 93.5, 88.4, 94.2, 91.8, 96.5, 90.3, 91.7, 93.8]

    x = np.arange(len(subjects))
    width = 0.25

    ax1.bar(x - width, base_scores, width, label='Base Model (GPT-3.5 size)',
            color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.2)
    ax1.bar(x, finetuned_scores, width, label='Fine-tuned Model',
            color='#27ae60', alpha=0.85, edgecolor='black', linewidth=1.2)
    ax1.bar(x + width, human_scores, width, label='Human Expert',
            color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.2)

    ax1.set_xlabel('MMLU Subject Category', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=13)
    ax1.set_title('MMLU Benchmark: Subject-Level Performance Breakdown', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(subjects, fontsize=11)
    ax1.set_ylim(0, 105)
    ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=12, loc='upper left')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add percentage improvement labels
    for i, (b, f) in enumerate(zip(base_scores, finetuned_scores)):
        improvement = f - b
        ax1.text(i, f + 2, f'+{improvement:.1f}%', ha='center',
                fontsize=9, fontweight='bold', color='green')

    # Panel 2: HumanEval Pass@k Analysis
    ax2 = fig.add_subplot(gs[1, 0])

    k_values = ['Pass@1', 'Pass@5', 'Pass@10', 'Pass@20', 'Pass@50', 'Pass@100']
    codex_scores = [28.5, 42.8, 52.3, 61.2, 71.8, 78.5]
    finetuned_scores = [42.1, 58.6, 67.9, 75.4, 83.2, 88.7]

    x = np.arange(len(k_values))
    width = 0.35

    ax2.bar(x - width/2, codex_scores, width, label='Codex (Base)',
            color='#e67e22', alpha=0.85, edgecolor='black', linewidth=1.2)
    ax2.bar(x + width/2, finetuned_scores, width, label='Fine-tuned on Code',
            color='#9b59b6', alpha=0.85, edgecolor='black', linewidth=1.2)

    ax2.set_xlabel('Pass@k Metric', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=12)
    ax2.set_title('HumanEval: Code Generation Success Rate', fontweight='bold', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(k_values, fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add annotation
    ax2.annotate('Pass@k: Probability of generating\nat least one correct solution\nin k attempts',
                xy=(4, 83), xytext=(2.5, 90),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=2))

    # Panel 3: TruthfulQA Categories
    ax3 = fig.add_subplot(gs[1, 1])

    categories = ['Science', 'Politics', 'Health', 'Conspiracy', 'Fiction', 'Myths']
    truthful_base = [52.3, 48.7, 45.2, 38.9, 51.8, 44.3]
    truthful_finetuned = [68.5, 62.4, 71.2, 58.3, 66.9, 61.7]

    x = np.arange(len(categories))
    width = 0.35

    ax3.bar(x - width/2, truthful_base, width, label='Base Model',
            color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.2)
    ax3.bar(x + width/2, truthful_finetuned, width, label='RLHF Fine-tuned',
            color='#1abc9c', alpha=0.85, edgecolor='black', linewidth=1.2)

    ax3.set_xlabel('Question Category', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Truthfulness Score (%)', fontweight='bold', fontsize=12)
    ax3.set_title('TruthfulQA: Factual Accuracy by Category', fontweight='bold', fontsize=13)
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=11)
    ax3.set_ylim(0, 85)
    ax3.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.axhline(y=50, color='orange', linestyle='--', linewidth=2, alpha=0.7,
               label='Random Baseline')

    plt.suptitle('Benchmark Performance: MMLU, HumanEval, TruthfulQA',
                 fontsize=16, fontweight='bold', y=0.995)

    output_path = os.path.join(OUTPUT_DIR, "benchmark_performance.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n✓ Saved: {output_path}")
    return output_path

if __name__ == "__main__":
    create_benchmark_performance()
    print("\n" + "=" * 80)
    print("VISUALIZATION 4/8 COMPLETE")
    print("=" * 80)
    print("\nNext visualization: evaluation_human_eval.py")
