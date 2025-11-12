"""
Visualization 7 of 8 for Section 14: Retraining Decision Framework

This script visualizes:
- Retraining decision logic (accuracy drop OR drift threshold)
- Cost-benefit analysis of retraining timing

Output: visualization/images/llmops/retraining_decision.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

OUTPUT_DIR = "../../images/llmops"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("SECTION 14 - VISUALIZATION 7/8: Retraining Decision Framework")
print("=" * 80)

def create_retraining_decision():
    """Two-panel visualization: retraining logic and cost-benefit analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Automated Retraining: Decision Logic & Cost-Benefit Trade-offs',
                 fontsize=16, fontweight='bold', y=1.02)

    # Panel 1: Retraining Decision Logic
    ax1 = axes[0]

    # Simulate 180 days of monitoring
    days = np.arange(1, 181)
    baseline_acc = 85.0

    # Accuracy degradation
    acc_drop = -0.03 * days + np.random.normal(0, 0.3, len(days))
    accuracy = baseline_acc + acc_drop
    accuracy = np.clip(accuracy, 75, 86)

    # KL divergence increase
    kl_drift = 0.001 * days + np.random.normal(0, 0.005, len(days))
    kl_drift = np.clip(kl_drift, 0, 0.2)

    # Create dual-axis plot
    ax1_twin = ax1.twinx()

    # Plot accuracy
    line1 = ax1.plot(days, accuracy, linewidth=2.5, color='#3498db',
                    label='Accuracy', alpha=0.85)

    # Plot KL divergence
    line2 = ax1_twin.plot(days, kl_drift, linewidth=2.5, color='#e67e22',
                         label='KL Divergence', alpha=0.85)

    # Accuracy thresholds
    ax1.axhline(y=baseline_acc, color='green', linestyle='--',
               linewidth=2, alpha=0.5, label='Baseline')
    ax1.axhline(y=baseline_acc * 0.95, color='orange', linestyle='--',
               linewidth=2, alpha=0.6, label='Warning (-5%)')
    ax1.axhline(y=baseline_acc * 0.90, color='red', linestyle='--',
               linewidth=2.5, alpha=0.7, label='Retrain Trigger (-10%)')

    # KL thresholds
    ax1_twin.axhline(y=0.1, color='red', linestyle=':', linewidth=2.5,
                    alpha=0.7, label='D_KL Threshold')

    # Mark retraining points
    # Point 1: Accuracy trigger (day 120)
    retrain_day_acc = 120
    ax1.plot(retrain_day_acc, accuracy[retrain_day_acc-1], 'r*',
            markersize=25, markeredgecolor='black', markeredgewidth=2, zorder=10)
    ax1.annotate('RETRAIN #1\nAccuracy Drop\n> 5% threshold',
                xy=(retrain_day_acc, accuracy[retrain_day_acc-1]),
                xytext=(80, 77),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    # Point 2: KL drift trigger (day 150)
    retrain_day_kl = 150
    ax1_twin.plot(retrain_day_kl, kl_drift[retrain_day_kl-1], 'r*',
                 markersize=25, markeredgecolor='black', markeredgewidth=2, zorder=10)
    ax1_twin.annotate('RETRAIN #2\nDrift > 0.1\nAND worsening',
                     xy=(retrain_day_kl, kl_drift[retrain_day_kl-1]),
                     xytext=(140, 0.16),
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                     fontsize=9, ha='center',
                     arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    ax1.set_xlabel('Days Since Deployment', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12, color='#3498db')
    ax1_twin.set_ylabel('KL Divergence', fontweight='bold', fontsize=12, color='#e67e22')
    ax1.set_title('Retraining Trigger Logic: Dual-Condition Monitoring',
                  fontweight='bold', fontsize=13)

    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax1_twin.tick_params(axis='y', labelcolor='#e67e22')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines + [ax1.get_legend_handles_labels()[0][i] for i in range(3)],
              labels + ['Baseline', 'Warning (-5%)', 'Retrain Trigger (-10%)'],
              frameon=True, fancybox=True, shadow=True, fontsize=9, loc='upper right')

    # Add decision formula
    ax1.text(0.05, 0.05, 'RETRAIN IF:\n(Δ_acc > 5%) OR\n(D_KL > 0.1 AND trend↑)',
            transform=ax1.transAxes, ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5),
            fontsize=10, fontweight='bold', family='monospace')

    # Panel 2: Cost-Benefit Analysis
    ax2 = axes[1]

    # Different retraining frequencies
    frequencies = ['Never\nRetrain', '12 Months', '6 Months', '3 Months',
                  '1 Month', 'Weekly']
    months_between = [float('inf'), 12, 6, 3, 1, 0.25]

    # Cost calculation (24-month period)
    period_months = 24

    # Retraining cost (one-time)
    retrain_cost = 50000  # USD

    # Operational cost increase from degraded accuracy
    # Assumes: Lower accuracy → more support tickets, refunds, lost business
    base_monthly_ops = 10000  # USD baseline

    total_costs = []
    accuracy_losses = []

    for months_freq in months_between:
        if months_freq == float('inf'):
            # Never retrain: accuracy degrades continuously
            n_retrains = 0
            # Accuracy drops 0.5% per month, revenue loss increases
            avg_acc_loss = 0.5 * period_months / 2  # Average over period
            ops_cost = base_monthly_ops * period_months * (1 + avg_acc_loss/100 * 3)  # 3x multiplier
        else:
            # Retrain periodically
            n_retrains = int(period_months / months_freq)
            # Accuracy resets after each retrain
            avg_acc_loss = 0.5 * months_freq / 2  # Average loss between retrains
            ops_cost = base_monthly_ops * period_months * (1 + avg_acc_loss/100 * 3)

        total_cost = (n_retrains * retrain_cost) + ops_cost
        total_costs.append(total_cost / 1000)  # Convert to thousands
        accuracy_losses.append(avg_acc_loss)

    # Create grouped bar chart
    x = np.arange(len(frequencies))
    width = 0.35

    bars1 = ax2.bar(x - width/2, [c - (base_monthly_ops * period_months / 1000) for c in total_costs],
                    width, label='Extra Cost (Retraining + Degradation)',
                    color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)

    bars2 = ax2.bar(x + width/2, [base_monthly_ops * period_months / 1000] * len(frequencies),
                   width, label='Baseline Ops Cost',
                   color='#95a5a6', alpha=0.85, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, cost, acc_loss) in enumerate(zip(bars1, total_costs, accuracy_losses)):
        height = bar.get_height()
        total = cost
        ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'${total:.0f}K\n({acc_loss:.1f}% loss)',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Highlight optimal
    optimal_idx = 2  # 6 months
    ax2.plot(optimal_idx, total_costs[optimal_idx], 'g*', markersize=25,
            markeredgecolor='black', markeredgewidth=2, zorder=10)
    ax2.annotate('OPTIMAL\nBalance',
                xy=(optimal_idx, total_costs[optimal_idx]),
                xytext=(optimal_idx, total_costs[optimal_idx] + 100),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                fontsize=10, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))

    ax2.set_xlabel('Retraining Frequency', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Total Cost Over 24 Months (USD thousands)', fontweight='bold', fontsize=12)
    ax2.set_title('Cost-Benefit Analysis: Finding Optimal Retraining Cadence',
                  fontweight='bold', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(frequencies, fontsize=10)
    ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add cost breakdown note
    ax2.text(0.05, 0.95, 'Cost Model:\n' +
             '• Retraining: $50K/event\n' +
             '• Ops degradation: 3× multiplier\n' +
             '• Accuracy loss: 0.5%/month\n' +
             '\nSweet spot: 3-6 months',
             transform=ax2.transAxes, ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
             fontsize=8)

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/retraining_decision.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    create_retraining_decision()
    print(f"\n✓ Section 14 Visualization 7/8 completed!")
    print(f"✓ Output: {OUTPUT_DIR}/retraining_decision.png")
