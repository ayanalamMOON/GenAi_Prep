"""
Section 11: Safety, Ethics, and Bias Mitigation - Mitigation Strategies Visualizations
Creates 2 visualizations for debiasing methods and their effectiveness
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
OUTPUT_DIR = "../images/safety"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# High-resolution publication quality
DPI = 300

def create_mitigation_strategies_visualization():
    """
    Create 2-panel visualization for mitigation strategies:
    1. Comparison of debiasing techniques effectiveness
    2. Bias reduction vs performance trade-off across methods
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Bias Mitigation Strategies: Effectiveness and Performance Trade-offs', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Panel 1: Debiasing Techniques Comparison
    ax1 = axes[1]
    
    # Different debiasing methods
    methods = ['No\nDebiasing', 'Data\nRebalancing', 'Counterfactual\nAugmentation', 
               'Adversarial\nDebiasing', 'Fair\nRepresentation', 'INLP\nProjection']
    
    # Metrics: Bias reduction (higher is better), Accuracy retention (higher is better)
    bias_reduction = [0, 35, 52, 68, 71, 78]  # Percentage reduction
    accuracy_retention = [100, 98, 96, 92, 90, 88]  # Percentage of original accuracy
    
    x = np.arange(len(methods))
    width = 0.35
    
    # Create grouped bar chart
    bars1 = ax1.bar(x - width/2, bias_reduction, width, label='Bias Reduction (%)',
                   color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, accuracy_retention, width, label='Accuracy Retention (%)',
                   color='#E67E22', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.0f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9)
    ax1.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(2) Debiasing Techniques Comparison\nBias Reduction vs Accuracy Retention', 
                  fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='lower left', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 110])
    
    # Add target thresholds
    ax1.axhline(y=50, color='red', linestyle='--', linewidth=2, 
               label='Target: 50% bias reduction', alpha=0.5)
    ax1.axhline(y=95, color='green', linestyle='--', linewidth=2,
               label='Target: 95% accuracy retention', alpha=0.5)
    ax1.legend(loc='lower left', fontsize=8)
    
    # Panel 2: Bias Reduction vs Performance Trade-off
    ax2 = axes[0]
    
    # Scatter plot showing trade-off
    # X-axis: Bias reduction percentage
    # Y-axis: Final accuracy percentage
    
    # Original accuracy (no debiasing)
    original_acc = 87.5
    
    # Calculate final accuracies based on retention
    final_accuracies = [original_acc * (ret/100) for ret in accuracy_retention]
    
    # Plot trajectory
    ax2.plot(bias_reduction, final_accuracies, 'o-', color='#8E44AD', linewidth=3,
            markersize=12, markeredgecolor='black', markeredgewidth=2,
            label='Mitigation Trajectory')
    
    # Annotate each method
    for i, method in enumerate(methods):
        # Adjust text position to avoid overlap
        offset_x = 3 if i % 2 == 0 else -3
        offset_y = 0.3 if i % 2 == 0 else -0.3
        ha = 'left' if i % 2 == 0 else 'right'
        
        ax2.annotate(method.replace('\n', ' '),
                    xy=(bias_reduction[i], final_accuracies[i]),
                    xytext=(bias_reduction[i] + offset_x, final_accuracies[i] + offset_y),
                    fontsize=8, fontweight='bold', ha=ha,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.4),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Highlight sweet spot
    sweet_spot_idx = 3  # Adversarial Debiasing
    ax2.plot([bias_reduction[sweet_spot_idx]], [final_accuracies[sweet_spot_idx]], 
            'g*', markersize=25, markeredgecolor='darkgreen', markeredgewidth=2,
            label='Recommended\nSweet Spot', zorder=10)
    
    # Add shaded acceptable region
    ax2.axvspan(50, 100, alpha=0.1, color='green', label='Target Bias\nReduction (>50%)')
    ax2.axhspan(original_acc * 0.90, original_acc, alpha=0.1, color='blue',
               label='Acceptable Accuracy\n(>90% original)')
    
    ax2.set_xlabel('Bias Reduction (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Final Model Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('(1) Bias-Accuracy Trade-off Landscape\nChoosing the Right Debiasing Method', 
                  fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-5, 85])
    ax2.set_ylim([75, 89])
    
    # Add interpretation box
    textstr = 'Sweet Spot:\n68% bias reduction\n92% accuracy retention\n80.5% final accuracy'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
    ax2.text(0.02, 0.02, textstr, transform=ax2.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='left', bbox=props,
            fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/mitigation_strategies_comparison.png", dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/mitigation_strategies_comparison.png")
    plt.close()

if __name__ == "__main__":
    print("Generating Section 11 visualizations: Mitigation Strategies...")
    create_mitigation_strategies_visualization()
    print("\n✓ All mitigation strategies visualizations generated successfully!")
    print(f"✓ Output directory: {OUTPUT_DIR}")
