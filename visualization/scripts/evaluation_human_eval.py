"""
Visualization 5 of 8 for Section 10: Human Evaluation Methods

This script visualizes human evaluation approaches:
- Pairwise preference comparison
- Likert scale ratings across dimensions
- Inter-annotator agreement
- Evaluation cost and sample size analysis

Output: visualization/images/evaluation/human_evaluation_methods.png
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

OUTPUT_DIR = "../images/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("SECTION 10 - VISUALIZATION 5/8: Human Evaluation Methods")
print("=" * 80)

def create_human_evaluation_methods():
    """Four-panel human evaluation analysis"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # Panel 1: Pairwise Preference Win Rates
    ax1 = fig.add_subplot(gs[0, 0])

    comparisons = ['vs Base\nModel', 'vs GPT-3.5', 'vs GPT-4', 'vs Human\nExpert']
    win_rate = [82.5, 68.3, 42.8, 28.5]
    tie_rate = [12.3, 18.7, 31.2, 25.8]
    lose_rate = [5.2, 13.0, 26.0, 45.7]

    x = np.arange(len(comparisons))

    p1 = ax1.barh(x, win_rate, label='Win', color='#27ae60', alpha=0.85, edgecolor='black', linewidth=1.2)
    p2 = ax1.barh(x, tie_rate, left=win_rate, label='Tie', color='#f39c12', alpha=0.85, edgecolor='black', linewidth=1.2)
    p3 = ax1.barh(x, lose_rate, left=[w+t for w,t in zip(win_rate, tie_rate)],
                  label='Lose', color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.2)

    ax1.set_yticks(x)
    ax1.set_yticklabels(comparisons, fontsize=11)
    ax1.set_xlabel('Percentage (%)', fontweight='bold', fontsize=12)
    ax1.set_title('Pairwise Preference: Fine-tuned Model Performance', fontweight='bold', fontsize=13)
    ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=11, loc='upper right')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 100)

    # Add percentage labels
    for i, (w, t, l) in enumerate(zip(win_rate, tie_rate, lose_rate)):
        ax1.text(w/2, i, f'{w:.1f}%', ha='center', va='center',
                color='white', fontsize=10, fontweight='bold')
        ax1.text(w + t/2, i, f'{t:.1f}%', ha='center', va='center',
                color='black', fontsize=10, fontweight='bold')
        ax1.text(w + t + l/2, i, f'{l:.1f}%', ha='center', va='center',
                color='white', fontsize=10, fontweight='bold')

    # Panel 2: Likert Scale Ratings (1-5)
    ax2 = fig.add_subplot(gs[0, 1])

    dimensions = ['Helpfulness', 'Harmlessness', 'Honesty', 'Coherence', 'Fluency', 'Relevance']
    base_ratings = [2.8, 3.1, 2.9, 3.4, 3.8, 3.2]
    finetuned_ratings = [4.2, 4.5, 4.1, 4.6, 4.7, 4.4]
    human_ratings = [4.8, 4.9, 4.7, 4.9, 4.9, 4.8]

    x = np.arange(len(dimensions))
    width = 0.25

    ax2.bar(x - width, base_ratings, width, label='Base Model',
            color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.2)
    ax2.bar(x, finetuned_ratings, width, label='Fine-tuned',
            color='#27ae60', alpha=0.85, edgecolor='black', linewidth=1.2)
    ax2.bar(x + width, human_ratings, width, label='Human Baseline',
            color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.2)

    ax2.set_xlabel('Evaluation Dimension', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Average Rating (1-5 scale)', fontweight='bold', fontsize=12)
    ax2.set_title('Likert Scale Ratings Across Dimensions', fontweight='bold', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(dimensions, rotation=45, ha='right', fontsize=10)
    ax2.set_ylim(0, 5.5)
    ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=3.0, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Acceptable Threshold')

    # Panel 3: Inter-Annotator Agreement Heatmap
    ax3 = fig.add_subplot(gs[1, 0])

    annotators = ['Ann. 1', 'Ann. 2', 'Ann. 3', 'Ann. 4', 'Ann. 5']

    # Simulate agreement matrix (Cohen's Kappa)
    np.random.seed(42)
    agreement_matrix = np.array([
        [1.00, 0.78, 0.72, 0.68, 0.75],
        [0.78, 1.00, 0.81, 0.74, 0.79],
        [0.72, 0.81, 1.00, 0.71, 0.76],
        [0.68, 0.74, 0.71, 1.00, 0.73],
        [0.75, 0.79, 0.76, 0.73, 1.00]
    ])

    im = ax3.imshow(agreement_matrix, cmap='RdYlGn', aspect='auto',
                    vmin=0.5, vmax=1.0, interpolation='nearest')

    ax3.set_xticks(np.arange(len(annotators)))
    ax3.set_yticks(np.arange(len(annotators)))
    ax3.set_xticklabels(annotators, fontsize=10)
    ax3.set_yticklabels(annotators, fontsize=10)
    ax3.set_title('Inter-Annotator Agreement (Cohen\'s Kappa)', fontweight='bold', fontsize=13)

    # Add agreement values
    for i in range(len(annotators)):
        for j in range(len(annotators)):
            value = agreement_matrix[i, j]
            color = 'white' if value < 0.7 else 'black'
            ax3.text(j, i, f'{value:.2f}', ha='center', va='center',
                    color=color, fontsize=10, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Agreement Score (0-1)', fontweight='bold', fontsize=11)

    # Add interpretation guide
    ax3.text(2.5, 5.5, 'Interpretation: >0.80 = Excellent, 0.60-0.80 = Good, <0.60 = Poor',
            ha='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

    # Panel 4: Evaluation Cost vs Sample Size
    ax4 = fig.add_subplot(gs[1, 1])

    sample_sizes = np.array([50, 100, 200, 500, 1000, 2000, 5000])
    cost_per_annotation = 2.5  # USD
    num_annotators = 3

    total_costs = sample_sizes * cost_per_annotation * num_annotators

    # Confidence intervals (95% CI width decreases with sqrt(n))
    ci_widths = 15 / np.sqrt(sample_sizes) * 100  # Percentage

    ax4_twin = ax4.twinx()

    # Plot cost on left axis
    line1 = ax4.plot(sample_sizes, total_costs, 'o-', linewidth=2.5, markersize=10,
                    color='#e74c3c', label='Total Cost', markeredgecolor='black', markeredgewidth=1.5)
    ax4.set_xlabel('Sample Size', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Total Cost (USD)', fontweight='bold', fontsize=12, color='#e74c3c')
    ax4.tick_params(axis='y', labelcolor='#e74c3c')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3, linestyle='--', which='both')

    # Plot CI width on right axis
    line2 = ax4_twin.plot(sample_sizes, ci_widths, 's-', linewidth=2.5, markersize=10,
                         color='#3498db', label='95% CI Width', markeredgecolor='black', markeredgewidth=1.5)
    ax4_twin.set_ylabel('Confidence Interval Width (%)', fontweight='bold', fontsize=12, color='#3498db')
    ax4_twin.tick_params(axis='y', labelcolor='#3498db')

    # Add combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=11)

    ax4.set_title('Evaluation Cost vs Statistical Confidence Trade-off', fontweight='bold', fontsize=13)

    # Add optimal point annotation
    optimal_idx = 3  # 500 samples
    ax4.annotate(f'Sweet Spot\n(n=500, ${total_costs[optimal_idx]:.0f})',
                xy=(sample_sizes[optimal_idx], total_costs[optimal_idx]),
                xytext=(sample_sizes[optimal_idx]*0.3, total_costs[optimal_idx]*1.5),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=2))

    plt.suptitle('Human Evaluation: Preferences, Ratings, Agreement, Cost Analysis',
                 fontsize=15, fontweight='bold', y=0.995)

    output_path = os.path.join(OUTPUT_DIR, "human_evaluation_methods.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n✓ Saved: {output_path}")
    return output_path

if __name__ == "__main__":
    create_human_evaluation_methods()
    print("\n" + "=" * 80)
    print("VISUALIZATION 5/8 COMPLETE")
    print("=" * 80)
    print("\nNext visualization: evaluation_error_analysis.py")
