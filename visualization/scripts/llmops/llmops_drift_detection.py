"""
Visualization 6 of 8 for Section 14: Drift Detection Methods

This script visualizes:
- KL divergence drift detection over time
- Population Stability Index (PSI) feature drift

Output: visualization/images/llmops/drift_detection.png
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
print("SECTION 14 - VISUALIZATION 6/8: Drift Detection Methods")
print("=" * 80)

def create_drift_detection():
    """Two-panel visualization: KL divergence and PSI drift detection"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Data Drift Detection: Monitoring Distribution Shifts',
                 fontsize=16, fontweight='bold', y=1.02)

    # Panel 1: KL Divergence Over Time
    ax1 = axes[0]

    # Simulate 12 weeks of KL divergence monitoring
    weeks = np.arange(1, 13)

    # Three drift scenarios
    # Scenario 1: Stable (minimal drift)
    kl_stable = 0.02 + np.random.normal(0, 0.01, len(weeks))
    kl_stable = np.clip(kl_stable, 0, 0.05)

    # Scenario 2: Gradual drift
    kl_gradual = 0.02 + 0.008 * weeks + np.random.normal(0, 0.01, len(weeks))
    kl_gradual = np.clip(kl_gradual, 0, 0.15)

    # Scenario 3: Sudden drift at week 7
    kl_sudden = np.concatenate([
        0.03 + np.random.normal(0, 0.005, 6),
        0.12 + np.random.normal(0, 0.01, 6)
    ])

    # Plot scenarios
    ax1.plot(weeks, kl_stable, marker='o', linewidth=2.5, markersize=8,
            color='#27ae60', label='Stable System', alpha=0.85)
    ax1.plot(weeks, kl_gradual, marker='s', linewidth=2.5, markersize=8,
            color='#f39c12', label='Gradual Drift', alpha=0.85)
    ax1.plot(weeks, kl_sudden, marker='^', linewidth=2.5, markersize=8,
            color='#e74c3c', label='Sudden Drift', alpha=0.85)

    # Threshold lines
    ax1.axhline(y=0.05, color='orange', linestyle='--', linewidth=2,
               label='Warning (D_KL = 0.05)', alpha=0.6)
    ax1.axhline(y=0.1, color='red', linestyle='--', linewidth=2.5,
               label='Critical (D_KL = 0.10)', alpha=0.7)

    # Highlight critical events
    ax1.plot(12, kl_gradual[-1], 'r*', markersize=20,
            markeredgecolor='black', markeredgewidth=2)
    ax1.annotate('Retrain Trigger\nD_KL = 0.11',
                xy=(12, kl_gradual[-1]), xytext=(9, 0.13),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    ax1.plot(7, kl_sudden[6], 'r*', markersize=20,
            markeredgecolor='black', markeredgewidth=2)
    ax1.annotate('Distribution Shift\nInvestigate!',
                xy=(7, kl_sudden[6]), xytext=(4.5, 0.14),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    ax1.set_xlabel('Weeks Since Deployment', fontweight='bold', fontsize=12)
    ax1.set_ylabel('KL Divergence D_KL(P_prod || P_train)', fontweight='bold', fontsize=12)
    ax1.set_title('KL Divergence: Token Distribution Drift Detection',
                  fontweight='bold', fontsize=13)
    ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 0.16)

    # Add formula
    ax1.text(0.98, 0.05, r'$D_{KL}(P||Q) = \sum P(x) \log \frac{P(x)}{Q(x)}$' + '\n' +
             'P = production dist.\nQ = training dist.',
             transform=ax1.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.4),
             fontsize=9, family='monospace')

    # Panel 2: Population Stability Index (PSI)
    ax2 = axes[1]

    # Feature drift heatmap using PSI
    features = ['Token Length', 'Sentiment Score', 'Entity Count',
                'Complexity', 'Domain Label', 'User Type']
    months = ['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6']

    # Generate PSI values (higher = more drift)
    np.random.seed(42)
    psi_values = np.random.rand(len(features), len(months)) * 0.3

    # Add some critical drift points
    psi_values[2, 3] = 0.35  # Entity Count drift in Month 4
    psi_values[4, 4] = 0.28  # Domain Label drift in Month 5
    psi_values[5, 5] = 0.42  # User Type drift in Month 6

    # Create heatmap
    im = ax2.imshow(psi_values, cmap='RdYlGn_r', aspect='auto',
                   vmin=0, vmax=0.5, alpha=0.9)

    # Add text annotations
    for i in range(len(features)):
        for j in range(len(months)):
            psi = psi_values[i, j]
            color = 'white' if psi > 0.25 else 'black'
            status = '⚠' if 0.1 < psi < 0.25 else '🚨' if psi >= 0.25 else '✓'
            ax2.text(j, i, f'{psi:.2f}\n{status}',
                    ha='center', va='center', color=color,
                    fontsize=9, fontweight='bold')

    # Set ticks and labels
    ax2.set_xticks(np.arange(len(months)))
    ax2.set_yticks(np.arange(len(features)))
    ax2.set_xticklabels(months, fontsize=10)
    ax2.set_yticklabels(features, fontsize=10)

    ax2.set_xlabel('Time Period', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Feature', fontweight='bold', fontsize=12)
    ax2.set_title('Population Stability Index (PSI): Feature-Level Drift',
                  fontweight='bold', fontsize=13)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('PSI Score', fontweight='bold', fontsize=11)

    # Add threshold annotations
    cbar.ax.axhline(y=0.1, color='orange', linestyle='--', linewidth=2)
    cbar.ax.axhline(y=0.25, color='red', linestyle='--', linewidth=2)
    cbar.ax.text(1.5, 0.1, '< 0.1: Stable', fontsize=8, va='center')
    cbar.ax.text(1.5, 0.175, '0.1-0.25: Warning', fontsize=8, va='center')
    cbar.ax.text(1.5, 0.35, '> 0.25: Critical', fontsize=8, va='center')

    # Add PSI interpretation
    ax2.text(0.5, -0.18, 'PSI Interpretation: ✓ Stable | ⚠ Monitor | 🚨 Investigate & Retrain\n' +
             'Calculates distribution shift per feature using binned histograms',
             transform=ax2.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
             fontsize=9, style='italic')

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/drift_detection.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    create_drift_detection()
    print(f"\n✓ Section 14 Visualization 6/8 completed!")
    print(f"✓ Output: {OUTPUT_DIR}/drift_detection.png")
