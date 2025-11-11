"""
Section 11: Safety, Ethics, and Bias Mitigation - Toxicity & Safety Visualizations
Creates 2 visualizations for toxicity detection and safety filtering
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
OUTPUT_DIR = "../../images/safety"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# High-resolution publication quality
DPI = 300

def create_toxicity_safety_visualization():
    """
    Create 2-panel visualization for toxicity and safety:
    1. Toxicity scores distribution across content categories
    2. Multi-layer safety filter effectiveness
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Toxicity Detection and Safety Filtering Effectiveness', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Panel 1: Toxicity Score Distribution
    ax1 = axes[0]
    
    # Content categories with toxicity distributions
    categories = ['Safe\nContent', 'Borderline\nContent', 'Toxic\nContent', 
                  'Hate\nSpeech', 'Threats']
    
    # Generate simulated toxicity score distributions
    np.random.seed(42)
    
    # Each category has different distribution
    safe_scores = np.random.beta(2, 10, 1000) * 0.3  # Low scores
    borderline_scores = np.random.beta(5, 5, 1000) * 0.7 + 0.15  # Middle
    toxic_scores = np.random.beta(10, 3, 1000) * 0.5 + 0.4  # High
    hate_scores = np.random.beta(15, 2, 1000) * 0.4 + 0.55  # Very high
    threats_scores = np.random.beta(20, 2, 1000) * 0.3 + 0.65  # Extreme
    
    data = [safe_scores, borderline_scores, toxic_scores, hate_scores, threats_scores]
    colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    
    # Create violin plots
    parts = ax1.violinplot(data, positions=range(len(categories)), 
                          showmeans=True, showmedians=True, widths=0.7)
    
    # Color the violin plots
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Customize other elements
    parts['cmeans'].set_color('blue')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('red')
    parts['cmedians'].set_linewidth(2)
    
    # Add toxicity thresholds
    ax1.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, 
               label='Moderate Toxicity (0.5)', alpha=0.7)
    ax1.axhline(y=0.8, color='red', linestyle='--', linewidth=2,
               label='High Toxicity (0.8)', alpha=0.7)
    
    # Add mean values as text
    means = [np.mean(d) for d in data]
    for i, mean_val in enumerate(means):
        ax1.text(i, mean_val + 0.05, f'μ={mean_val:.2f}', 
                ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, fontsize=10)
    ax1.set_ylabel('Toxicity Score (0-1)', fontsize=11, fontweight='bold')
    ax1.set_title('(1) Toxicity Score Distribution\nAcross Content Categories', 
                  fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([-0.05, 1.05])
    
    # Add interpretation box
    textstr = 'Score Range:\n0.0-0.3: Safe\n0.3-0.5: Borderline\n0.5-0.8: Toxic\n0.8-1.0: Severe'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.3)
    ax1.text(0.98, 0.98, textstr, transform=ax1.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Panel 2: Multi-layer Safety Filter Effectiveness
    ax2 = axes[1]
    
    # Layers of defense
    layers = ['Input\nRaw', 'Input\nFilter', 'Content\nPolicy', 'Model\nOutput', 
              'Output\nFilter', 'Final\nOutput']
    
    # Percentage of unsafe content remaining at each layer
    # Start with 100% unsafe, progressively filter
    unsafe_remaining = [100, 78, 45, 28, 12, 3]
    safe_remaining = [100 - x for x in unsafe_remaining]
    
    x = np.arange(len(layers))
    width = 0.6
    
    # Stacked bar chart
    bars1 = ax2.bar(x, safe_remaining, width, label='Safe Content', 
                   color='#2ECC71', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x, unsafe_remaining, width, bottom=safe_remaining,
                   label='Unsafe Content', color='#E74C3C', alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    
    # Add percentage labels
    for i, (safe, unsafe) in enumerate(zip(safe_remaining, unsafe_remaining)):
        # Safe label
        if safe > 5:
            ax2.text(i, safe/2, f'{safe:.0f}%', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
        # Unsafe label
        if unsafe > 5:
            ax2.text(i, safe + unsafe/2, f'{unsafe:.0f}%', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
    
    # Add reduction arrows
    for i in range(len(layers) - 1):
        if unsafe_remaining[i] > unsafe_remaining[i+1]:
            reduction = unsafe_remaining[i] - unsafe_remaining[i+1]
            mid_y = safe_remaining[i] + unsafe_remaining[i] + 10
            ax2.annotate(f'-{reduction:.0f}%', xy=(i+0.5, mid_y),
                        fontsize=9, ha='center', fontweight='bold', color='red',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers, fontsize=10)
    ax2.set_ylabel('Content Percentage (%)', fontsize=11, fontweight='bold')
    ax2.set_title('(2) Multi-Layer Safety Filter Pipeline\nProgressive Threat Reduction', 
                  fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 125])
    
    # Add effectiveness metric
    overall_reduction = unsafe_remaining[0] - unsafe_remaining[-1]
    effectiveness = (overall_reduction / unsafe_remaining[0]) * 100
    
    textstr = f'Overall Effectiveness:\n{effectiveness:.0f}% threat reduction\n({unsafe_remaining[0]:.0f}% → {unsafe_remaining[-1]:.0f}%)'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.4)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left', bbox=props,
            fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/toxicity_safety_analysis.png", dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/toxicity_safety_analysis.png")
    plt.close()

if __name__ == "__main__":
    print("Generating Section 11 visualizations: Toxicity & Safety...")
    create_toxicity_safety_visualization()
    print("\n✓ All toxicity & safety visualizations generated successfully!")
    print(f"✓ Output directory: {OUTPUT_DIR}")
