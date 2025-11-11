"""
Section 11: Safety, Ethics, and Bias Mitigation - Fairness Metrics Visualizations
Creates 2 visualizations for fairness metrics (ROC Curves and Fairness Trade-offs)
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

def create_fairness_metrics_visualization():
    """
    Create 2-panel visualization for fairness metrics:
    1. ROC curves showing equalized odds violation across groups
    2. Fairness-accuracy trade-off (Pareto frontier)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Fairness Metrics: Equalized Odds and Accuracy-Fairness Trade-offs', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Panel 1: ROC Curves for Two Groups (Equalized Odds Violation)
    ax1 = axes[0]
    
    # Generate ROC curves for two demographic groups
    fpr_values = np.linspace(0, 1, 100)
    
    # Group A (advantaged) - higher TPR at same FPR
    tpr_group_a = 1 - np.exp(-5 * fpr_values)
    tpr_group_a = tpr_group_a / tpr_group_a[-1]  # Normalize to end at 1
    
    # Group B (disadvantaged) - lower TPR at same FPR
    tpr_group_b = 1 - np.exp(-3.5 * fpr_values)
    tpr_group_b = tpr_group_b / tpr_group_b[-1]
    
    # Plot ROC curves
    ax1.plot(fpr_values, fpr_values, 'k--', linewidth=2, alpha=0.5, label='Random Classifier')
    ax1.plot(fpr_values, tpr_group_a, 'b-', linewidth=3, label='Group A (AUC=0.93)', marker='o', markevery=10)
    ax1.plot(fpr_values, tpr_group_b, 'r-', linewidth=3, label='Group B (AUC=0.85)', marker='s', markevery=10)
    
    # Mark operating points
    op_fpr = 0.15
    op_tpr_a = 1 - np.exp(-5 * op_fpr)
    op_tpr_b = 1 - np.exp(-3.5 * op_fpr)
    
    ax1.plot([op_fpr], [op_tpr_a], 'bo', markersize=12, label=f'Op. Point A (FPR={op_fpr:.2f}, TPR={op_tpr_a:.2f})')
    ax1.plot([op_fpr], [op_tpr_b], 'rs', markersize=12, label=f'Op. Point B (FPR={op_fpr:.2f}, TPR={op_tpr_b:.2f})')
    
    # Draw arrow showing TPR gap
    ax1.annotate('', xy=(op_fpr, op_tpr_a), xytext=(op_fpr, op_tpr_b),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2.5))
    ax1.text(op_fpr + 0.02, (op_tpr_a + op_tpr_b) / 2, 
            f'ΔTPR = {op_tpr_a - op_tpr_b:.3f}\n(Equalized Odds\nViolation)',
            fontsize=9, color='purple', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (TPR)', fontsize=11, fontweight='bold')
    ax1.set_title('(1) ROC Curves: Equalized Odds Violation\nDifferent TPR at Same FPR Across Groups', 
                  fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    
    # Add ideal point
    ax1.plot([0], [1], 'g*', markersize=15, label='Perfect Classifier')
    
    # Panel 2: Fairness-Accuracy Trade-off (Pareto Frontier)
    ax2 = axes[1]
    
    # Simulate different fairness-accuracy combinations
    # X-axis: Fairness violation (demographic parity difference)
    # Y-axis: Model accuracy
    
    # Unconstrained model (high accuracy, high bias)
    fairness_violations = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    accuracies = np.array([0.92, 0.91, 0.89, 0.87, 0.85, 0.82, 0.78, 0.74, 0.70])
    
    # Plot Pareto frontier
    ax2.plot(fairness_violations, accuracies, 'o-', color='#2E86AB', linewidth=3, 
            markersize=10, label='Pareto Frontier', markeredgecolor='black', markeredgewidth=1.5)
    
    # Highlight specific points
    # Perfect fairness, lower accuracy
    ax2.plot([fairness_violations[0]], [accuracies[0]], 'gs', markersize=15,
            label=f'Fair Model (ΔDP={fairness_violations[0]:.2f}, Acc={accuracies[0]:.2f})',
            markeredgecolor='black', markeredgewidth=2)
    
    # Moderate fairness-accuracy balance
    mid_idx = 3
    ax2.plot([fairness_violations[mid_idx]], [accuracies[mid_idx]], 'yo', markersize=15,
            label=f'Balanced (ΔDP={fairness_violations[mid_idx]:.2f}, Acc={accuracies[mid_idx]:.2f})',
            markeredgecolor='black', markeredgewidth=2)
    
    # High accuracy, high bias
    ax2.plot([fairness_violations[-1]], [accuracies[-1]], 'r^', markersize=15,
            label=f'Biased Model (ΔDP={fairness_violations[-1]:.2f}, Acc={accuracies[-1]:.2f})',
            markeredgecolor='black', markeredgewidth=2)
    
    # Add annotations
    for i in [0, mid_idx, -1]:
        ax2.annotate(f'({fairness_violations[i]:.2f}, {accuracies[i]:.2f})',
                    xy=(fairness_violations[i], accuracies[i]),
                    xytext=(fairness_violations[i] + 0.02, accuracies[i] - 0.03),
                    fontsize=8, fontweight='bold')
    
    # Add fairness threshold line
    fairness_threshold = 0.10
    ax2.axvline(x=fairness_threshold, color='red', linestyle='--', linewidth=2,
               label=f'Fairness Threshold (ΔDP<{fairness_threshold:.2f})', alpha=0.7)
    
    # Shade acceptable region
    ax2.axvspan(0, fairness_threshold, alpha=0.1, color='green', 
               label='Acceptable Fairness Region')
    
    ax2.set_xlabel('Demographic Parity Violation (ΔDP)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Model Accuracy', fontsize=11, fontweight='bold')
    ax2.set_title('(2) Fairness-Accuracy Trade-off\nPareto Frontier Analysis', 
                  fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-0.02, 0.45])
    ax2.set_ylim([0.68, 0.94])
    
    # Add interpretation box
    textstr = 'Interpretation:\n• Lower ΔDP = Fairer\n• Higher Acc = Better\n• No free lunch:\n  Can\'t maximize both'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
    ax2.text(0.98, 0.98, textstr, transform=ax2.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fairness_metrics_analysis.png", dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/fairness_metrics_analysis.png")
    plt.close()

if __name__ == "__main__":
    print("Generating Section 11 visualizations: Fairness Metrics...")
    create_fairness_metrics_visualization()
    print("\n✓ All fairness metrics visualizations generated successfully!")
    print(f"✓ Output directory: {OUTPUT_DIR}")
