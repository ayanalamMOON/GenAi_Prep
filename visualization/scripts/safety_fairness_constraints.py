"""
Section 11: Safety, Ethics, and Bias Mitigation - Fairness Constraints Visualizations
Creates 2 visualizations for detailed fairness constraint analysis
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

def create_fairness_constraints_visualization():
    """
    Create 2-panel visualization for fairness constraints:
    1. Demographic parity analysis across multiple groups
    2. Equalized odds detailed breakdown (TPR and FPR comparison)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Fairness Constraints: Demographic Parity and Equalized Odds Analysis', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Panel 1: Demographic Parity Across Multiple Groups
    ax1 = axes[0]
    
    # Different demographic groups
    groups = ['White\nMale', 'White\nFemale', 'Black\nMale', 'Black\nFemale',
             'Asian\nMale', 'Asian\nFemale', 'Hispanic\nMale', 'Hispanic\nFemale']
    
    # Positive prediction rates (percentage receiving positive predictions)
    # Fair model should have similar rates across groups
    prediction_rates_biased = [58.5, 52.3, 38.2, 35.7, 48.9, 44.1, 42.3, 39.8]
    prediction_rates_fair = [47.2, 46.8, 46.5, 47.1, 46.3, 47.5, 46.9, 47.3]
    
    x = np.arange(len(groups))
    width = 0.35
    
    # Create grouped bar chart
    bars1 = ax1.bar(x - width/2, prediction_rates_biased, width, label='Biased Model',
                   color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, prediction_rates_fair, width, label='Debiased Model',
                   color='#27AE60', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    
    # Add fairness threshold band
    mean_rate = np.mean(prediction_rates_fair)
    ax1.axhspan(mean_rate - 2, mean_rate + 2, alpha=0.2, color='green',
               label=f'Fairness Band (±2%)')
    ax1.axhline(y=mean_rate, color='green', linestyle='--', linewidth=2, alpha=0.5)
    
    # Calculate demographic parity violation
    dp_violation_biased = max(prediction_rates_biased) - min(prediction_rates_biased)
    dp_violation_fair = max(prediction_rates_fair) - min(prediction_rates_fair)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, fontsize=9, rotation=0)
    ax1.set_ylabel('Positive Prediction Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(1) Demographic Parity Analysis\nPositive Prediction Rates Across Groups', 
                  fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([30, 65])
    
    # Add annotation box with ΔDP
    textstr = f'Demographic Parity Violation (ΔDP):\n'
    textstr += f'• Biased Model: {dp_violation_biased:.1f}%\n'
    textstr += f'• Debiased Model: {dp_violation_fair:.1f}%\n'
    textstr += f'• Improvement: {dp_violation_biased - dp_violation_fair:.1f}%'
    
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='black', linewidth=2)
    ax1.text(0.02, 0.02, textstr, transform=ax1.transAxes, fontsize=8,
            verticalalignment='bottom', horizontalalignment='left', bbox=props,
            fontweight='bold')
    
    # Panel 2: Equalized Odds - TPR and FPR Breakdown
    ax2 = axes[1]
    
    # Four demographic groups for clarity
    groups_eo = ['White', 'Black', 'Asian', 'Hispanic']
    
    # True Positive Rates (sensitivity) - should be equal across groups
    tpr_biased = [0.82, 0.68, 0.76, 0.71]
    tpr_fair = [0.78, 0.77, 0.79, 0.78]
    
    # False Positive Rates (1 - specificity) - should be equal across groups
    fpr_biased = [0.12, 0.24, 0.16, 0.20]
    fpr_fair = [0.15, 0.16, 0.14, 0.15]
    
    # Create grouped bar chart with TPR and FPR side by side
    x = np.arange(len(groups_eo))
    width = 0.2
    
    # TPR bars
    bars1 = ax2.bar(x - 1.5*width, tpr_biased, width, label='TPR (Biased)',
                   color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x - 0.5*width, tpr_fair, width, label='TPR (Fair)',
                   color='#1ABC9C', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # FPR bars
    bars3 = ax2.bar(x + 0.5*width, fpr_biased, width, label='FPR (Biased)',
                   color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars4 = ax2.bar(x + 1.5*width, fpr_fair, width, label='FPR (Fair)',
                   color='#F39C12', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Add ideal reference lines
    ax2.axhline(y=0.78, color='#1ABC9C', linestyle='--', linewidth=2, alpha=0.4,
               label='Target TPR (0.78)')
    ax2.axhline(y=0.15, color='#F39C12', linestyle='--', linewidth=2, alpha=0.4,
               label='Target FPR (0.15)')
    
    # Calculate equalized odds violations
    tpr_violation_biased = max(tpr_biased) - min(tpr_biased)
    tpr_violation_fair = max(tpr_fair) - min(tpr_fair)
    fpr_violation_biased = max(fpr_biased) - min(fpr_biased)
    fpr_violation_fair = max(fpr_fair) - min(fpr_fair)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups_eo, fontsize=10)
    ax2.set_ylabel('Rate', fontsize=11, fontweight='bold')
    ax2.set_title('(2) Equalized Odds Analysis\nTPR and FPR Breakdown Across Groups', 
                  fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 0.95])
    
    # Add annotation box with violations
    textstr = f'Equalized Odds Violations:\n'
    textstr += f'ΔTPR (Biased): {tpr_violation_biased:.3f}\n'
    textstr += f'ΔTPR (Fair): {tpr_violation_fair:.3f}\n'
    textstr += f'ΔFPR (Biased): {fpr_violation_biased:.3f}\n'
    textstr += f'ΔFPR (Fair): {fpr_violation_fair:.3f}\n'
    textstr += f'\nBoth must be ≈0 for fairness'
    
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, edgecolor='black', linewidth=2)
    ax2.text(0.97, 0.97, textstr, transform=ax2.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right', bbox=props,
            fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fairness_constraints_detailed.png", dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/fairness_constraints_detailed.png")
    plt.close()

if __name__ == "__main__":
    print("Generating Section 11 visualizations: Fairness Constraints...")
    create_fairness_constraints_visualization()
    print("\n✓ All fairness constraints visualizations generated successfully!")
    print(f"✓ Output directory: {OUTPUT_DIR}")
