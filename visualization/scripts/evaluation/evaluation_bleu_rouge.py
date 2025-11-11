"""
Visualization 3 of 8 for Section 10: BLEU/ROUGE Metrics Analysis

This script visualizes overlap-based metrics:
- BLEU score breakdown (1-gram to 4-gram)
- ROUGE variants comparison (ROUGE-1, ROUGE-2, ROUGE-L)
- Precision vs Recall trade-off
- Reference text alignment visualization

Output: visualization/images/evaluation/bleu_rouge_analysis.png
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

OUTPUT_DIR = "../../images/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("SECTION 10 - VISUALIZATION 3/8: BLEU/ROUGE Metrics Analysis")
print("=" * 80)

def create_bleu_rouge_analysis():
    """Four-panel BLEU/ROUGE deep dive"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel 1: BLEU Score N-gram Breakdown
    ax1 = fig.add_subplot(gs[0, 0])

    models = ['Base\nModel', 'Fine-tuned\n(1K samples)', 'Fine-tuned\n(10K samples)', 'Fine-tuned\n(100K samples)']
    bleu1 = [0.52, 0.61, 0.68, 0.72]
    bleu2 = [0.38, 0.48, 0.56, 0.62]
    bleu3 = [0.28, 0.38, 0.47, 0.54]
    bleu4 = [0.21, 0.31, 0.40, 0.48]

    x = np.arange(len(models))
    width = 0.2

    ax1.bar(x - 1.5*width, bleu1, width, label='BLEU-1', color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.2)
    ax1.bar(x - 0.5*width, bleu2, width, label='BLEU-2', color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.2)
    ax1.bar(x + 0.5*width, bleu3, width, label='BLEU-3', color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1.2)
    ax1.bar(x + 1.5*width, bleu4, width, label='BLEU-4', color='#f39c12', alpha=0.85, edgecolor='black', linewidth=1.2)

    ax1.set_xlabel('Model Type', fontweight='bold', fontsize=12)
    ax1.set_ylabel('BLEU Score (0-1)', fontweight='bold', fontsize=12)
    ax1.set_title('BLEU N-gram Precision Breakdown', fontweight='bold', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=11, loc='upper left')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 0.85)

    # Panel 2: ROUGE Variants Comparison
    ax2 = fig.add_subplot(gs[0, 1])

    tasks = ['Summarization', 'Translation', 'Paraphrase', 'QA', 'Dialogue']
    rouge1_f1 = [0.52, 0.45, 0.58, 0.48, 0.51]
    rouge2_f1 = [0.38, 0.32, 0.42, 0.35, 0.38]
    rougeL_f1 = [0.48, 0.41, 0.54, 0.44, 0.47]

    x = np.arange(len(tasks))
    width = 0.25

    ax2.bar(x - width, rouge1_f1, width, label='ROUGE-1', color='#9b59b6', alpha=0.85, edgecolor='black', linewidth=1.2)
    ax2.bar(x, rouge2_f1, width, label='ROUGE-2', color='#1abc9c', alpha=0.85, edgecolor='black', linewidth=1.2)
    ax2.bar(x + width, rougeL_f1, width, label='ROUGE-L', color='#e67e22', alpha=0.85, edgecolor='black', linewidth=1.2)

    ax2.set_xlabel('Task Type', fontweight='bold', fontsize=12)
    ax2.set_ylabel('ROUGE F1 Score (0-1)', fontweight='bold', fontsize=12)
    ax2.set_title('ROUGE Variants Across Different Tasks', fontweight='bold', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks, fontsize=10)
    ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 0.7)

    # Panel 3: Precision vs Recall Trade-off
    ax3 = fig.add_subplot(gs[1, 0])

    # Generate precision-recall curves for different beam sizes
    recall_conservative = np.linspace(0.2, 0.6, 50)
    precision_conservative = 0.8 - 0.5 * recall_conservative + 0.1 * np.random.randn(50) * 0.05

    recall_balanced = np.linspace(0.3, 0.75, 50)
    precision_balanced = 0.75 - 0.4 * recall_balanced + 0.1 * np.random.randn(50) * 0.05

    recall_aggressive = np.linspace(0.4, 0.85, 50)
    precision_aggressive = 0.7 - 0.35 * recall_aggressive + 0.1 * np.random.randn(50) * 0.05

    ax3.plot(recall_conservative, precision_conservative, linewidth=2.5,
            label='Conservative (Beam=1)', color='#e74c3c', marker='o', markevery=10, markersize=8)
    ax3.plot(recall_balanced, precision_balanced, linewidth=2.5,
            label='Balanced (Beam=5)', color='#3498db', marker='s', markevery=10, markersize=8)
    ax3.plot(recall_aggressive, precision_aggressive, linewidth=2.5,
            label='Aggressive (Beam=10)', color='#2ecc71', marker='^', markevery=10, markersize=8)

    # Add F1 iso-lines
    for f1 in [0.4, 0.5, 0.6]:
        recall_line = np.linspace(0.1, 0.9, 100)
        precision_line = f1 / (2 * recall_line - f1 + 1e-10)
        precision_line = np.clip(precision_line, 0, 1)
        ax3.plot(recall_line, precision_line, '--', alpha=0.3, color='gray', linewidth=1.5)
        ax3.text(0.85, f1/(2*0.85 - f1), f'F1={f1:.1f}', fontsize=9, color='gray', fontweight='bold')

    ax3.set_xlabel('Recall', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Precision', fontweight='bold', fontsize=12)
    ax3.set_title('Precision-Recall Trade-off (BLEU/ROUGE)', fontweight='bold', fontsize=13)
    ax3.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    # Panel 4: N-gram Overlap Heatmap
    ax4 = fig.add_subplot(gs[1, 1])

    # Simulate n-gram overlap between generated and reference text
    ngrams = ['1-gram', '2-gram', '3-gram', '4-gram', '5-gram']
    samples = ['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5', 'Sample 6']

    # Create synthetic overlap data (higher values = better overlap)
    np.random.seed(42)
    overlap_data = np.array([
        [0.85, 0.72, 0.58, 0.42, 0.28],  # Sample 1 - good
        [0.78, 0.65, 0.48, 0.35, 0.22],  # Sample 2 - decent
        [0.92, 0.81, 0.69, 0.55, 0.38],  # Sample 3 - excellent
        [0.68, 0.52, 0.38, 0.24, 0.15],  # Sample 4 - mediocre
        [0.88, 0.75, 0.62, 0.48, 0.32],  # Sample 5 - very good
        [0.82, 0.68, 0.54, 0.40, 0.25],  # Sample 6 - good
    ])

    im = ax4.imshow(overlap_data, cmap='YlGnBu', aspect='auto', interpolation='nearest',
                    vmin=0, vmax=1)

    ax4.set_xticks(np.arange(len(ngrams)))
    ax4.set_yticks(np.arange(len(samples)))
    ax4.set_xticklabels(ngrams, fontsize=10)
    ax4.set_yticklabels(samples, fontsize=10)
    ax4.set_xlabel('N-gram Size', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Generated Samples', fontweight='bold', fontsize=12)
    ax4.set_title('N-gram Overlap with Reference Text', fontweight='bold', fontsize=13)

    # Add overlap percentages
    for i in range(len(samples)):
        for j in range(len(ngrams)):
            value = overlap_data[i, j]
            color = 'white' if value > 0.6 else 'black'
            ax4.text(j, i, f'{value:.2f}', ha='center', va='center',
                    color=color, fontsize=9, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label('Overlap Ratio (Higher = Better)', fontweight='bold', fontsize=11)

    plt.suptitle('BLEU/ROUGE Analysis: N-gram Precision, Variants, Precision-Recall, Overlap',
                 fontsize=15, fontweight='bold', y=0.995)

    output_path = os.path.join(OUTPUT_DIR, "bleu_rouge_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n✓ Saved: {output_path}")
    return output_path

if __name__ == "__main__":
    create_bleu_rouge_analysis()
    print("\n" + "=" * 80)
    print("VISUALIZATION 3/8 COMPLETE")
    print("=" * 80)
    print("\nNext visualization: evaluation_benchmarks.py")
