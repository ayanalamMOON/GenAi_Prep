"""
Visualization 2 of 8 for Section 10: Perplexity Analysis

This script generates detailed perplexity visualizations:
- Perplexity vs training steps
- Perplexity across different domains
- Perplexity distribution analysis
- Token-level surprise heatmap

Output: visualization/images/evaluation/perplexity_analysis.png
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

OUTPUT_DIR = "../images/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("SECTION 10 - VISUALIZATION 2/8: Perplexity Analysis")
print("=" * 80)

def create_perplexity_analysis():
    """Four-panel perplexity deep dive"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel 1: Perplexity vs Training Steps
    ax1 = fig.add_subplot(gs[0, 0])
    steps = np.arange(0, 10000, 100)
    ppl_train = 120 * np.exp(-steps / 2000) + 15
    ppl_val = 125 * np.exp(-steps / 2000) + 18
    ppl_test = 130 * np.exp(-steps / 2000) + 20

    ax1.plot(steps, ppl_train, linewidth=2.5, label='Training Set', color='#27ae60', marker='o', markevery=10)
    ax1.plot(steps, ppl_val, linewidth=2.5, label='Validation Set', color='#3498db', marker='s', markevery=10)
    ax1.plot(steps, ppl_test, linewidth=2.5, label='Test Set', color='#e74c3c', marker='^', markevery=10)

    ax1.set_xlabel('Training Steps', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Perplexity (Lower = Better)', fontweight='bold', fontsize=12)
    ax1.set_title('Perplexity Learning Curves', fontweight='bold', fontsize=13)
    ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')
    ax1.axhline(y=20, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Target PPL')

    # Panel 2: Perplexity Across Domains
    ax2 = fig.add_subplot(gs[0, 1])
    domains = ['General\nWeb', 'Medical', 'Legal', 'Code', 'Math', 'News', 'Social\nMedia']
    base_ppl = [25, 45, 52, 38, 48, 22, 28]
    finetuned_ppl = [23, 18, 20, 16, 21, 21, 26]

    x = np.arange(len(domains))
    width = 0.35
    bars1 = ax2.bar(x - width/2, base_ppl, width, label='Base Model',
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, finetuned_ppl, width, label='Fine-tuned',
                    color='#27ae60', alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add improvement percentages
    for i, (b, f) in enumerate(zip(base_ppl, finetuned_ppl)):
        improvement = ((b - f) / b) * 100
        ax2.text(i, max(b, f) + 2, f'-{improvement:.0f}%', ha='center',
                fontsize=9, fontweight='bold', color='green')

    ax2.set_xlabel('Domain', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Perplexity', fontweight='bold', fontsize=12)
    ax2.set_title('Domain-Specific Perplexity', fontweight='bold', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(domains, fontsize=10)
    ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Panel 3: Perplexity Distribution
    ax3 = fig.add_subplot(gs[1, 0])

    # Generate synthetic perplexity distributions
    np.random.seed(42)
    ppl_good_model = np.random.lognormal(mean=2.5, sigma=0.4, size=1000)
    ppl_bad_model = np.random.lognormal(mean=3.5, sigma=0.6, size=1000)

    ax3.hist(ppl_good_model, bins=50, alpha=0.7, label='Fine-tuned Model',
            color='#27ae60', edgecolor='black', linewidth=1.2, density=True)
    ax3.hist(ppl_bad_model, bins=50, alpha=0.7, label='Base Model',
            color='#e74c3c', edgecolor='black', linewidth=1.2, density=True)

    ax3.axvline(np.median(ppl_good_model), color='#27ae60', linestyle='--',
               linewidth=2.5, label=f'Median (Fine-tuned): {np.median(ppl_good_model):.1f}')
    ax3.axvline(np.median(ppl_bad_model), color='#e74c3c', linestyle='--',
               linewidth=2.5, label=f'Median (Base): {np.median(ppl_bad_model):.1f}')

    ax3.set_xlabel('Perplexity', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Density', fontweight='bold', fontsize=12)
    ax3.set_title('Perplexity Distribution Across Samples', fontweight='bold', fontsize=13)
    ax3.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(0, 100)

    # Panel 4: Token-Level Surprise Heatmap
    ax4 = fig.add_subplot(gs[1, 1])

    # Simulate token-level perplexity for a sentence
    tokens = ['The', 'patient', 'has', 'severe', 'dyspnea', 'and', 'requires', 'oxygen']
    positions = np.arange(len(tokens))

    # Simulate perplexity for each token (base vs fine-tuned)
    base_token_ppl = np.array([5, 25, 8, 35, 120, 15, 45, 80])
    finetuned_token_ppl = np.array([4, 12, 6, 15, 18, 10, 12, 15])

    # Create heatmap data
    heatmap_data = np.vstack([base_token_ppl, finetuned_token_ppl])

    im = ax4.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto', interpolation='nearest',
                    vmin=0, vmax=120)

    ax4.set_xticks(positions)
    ax4.set_xticklabels(tokens, rotation=45, ha='right', fontsize=10)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Base Model', 'Fine-tuned'], fontsize=11)
    ax4.set_title('Token-Level Perplexity (Medical Sentence)', fontweight='bold', fontsize=13)

    # Add perplexity values on heatmap
    for i in range(2):
        for j in range(len(tokens)):
            value = heatmap_data[i, j]
            color = 'white' if value > 60 else 'black'
            ax4.text(j, i, f'{value:.0f}', ha='center', va='center',
                    color=color, fontsize=10, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label('Perplexity (Lower = Better)', fontweight='bold', fontsize=11)

    plt.suptitle('Perplexity Analysis: Learning Curves, Domains, Distributions, Token-Level',
                 fontsize=15, fontweight='bold', y=0.995)

    output_path = os.path.join(OUTPUT_DIR, "perplexity_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n✓ Saved: {output_path}")
    return output_path

if __name__ == "__main__":
    create_perplexity_analysis()
    print("\n" + "=" * 80)
    print("VISUALIZATION 2/8 COMPLETE")
    print("=" * 80)
    print("\nNext visualization: evaluation_bleu_rouge.py")
