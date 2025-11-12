"""
Visualization 3 of 8 for Section 14: Hardware Failure & Checkpointing

This script visualizes:
- GPU failure probability over training duration
- Checkpointing frequency vs recovery time trade-off

Output: visualization/images/llmops/failure_checkpointing.png
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
print("SECTION 14 - VISUALIZATION 3/8: Failure Probability & Checkpointing")
print("=" * 80)

def create_failure_checkpointing():
    """Two-panel visualization: failure probability and checkpointing strategy"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Production Training: Hardware Failures & Recovery Strategy',
                 fontsize=16, fontweight='bold', y=1.02)

    # Panel 1: Failure Probability Over Time
    ax1 = axes[0]

    # Training duration in days
    days = np.linspace(0, 90, 200)

    # Different GPU cluster sizes
    gpu_configs = [
        (8, '#27ae60', '8 GPUs'),
        (16, '#3498db', '16 GPUs'),
        (32, '#f39c12', '32 GPUs'),
        (64, '#e67e22', '64 GPUs'),
        (128, '#e74c3c', '128 GPUs')
    ]

    # MTBF = 10 years = 3650 days
    mtbf = 3650

    for n_gpus, color, label in gpu_configs:
        # Failure rate: λ = N / MTBF
        lambda_system = n_gpus / mtbf

        # Probability of failure: P(t) = 1 - e^(-λt)
        prob_failure = 1 - np.exp(-lambda_system * days)

        ax1.plot(days, prob_failure * 100, linewidth=2.5,
                color=color, label=label, alpha=0.85)

    ax1.set_xlabel('Training Duration (Days)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Failure Probability (%)', fontweight='bold', fontsize=12)
    ax1.set_title('Hardware Failure Probability: Impact of Cluster Size\n(MTBF = 10 years per GPU)',
                  fontweight='bold', fontsize=13)
    ax1.set_xlim(0, 90)
    ax1.set_ylim(0, 100)
    ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Add reference lines
    ax1.axhline(y=50, color='orange', linestyle='--', linewidth=2, alpha=0.5)
    ax1.text(85, 52, '50%', fontsize=10, color='orange', fontweight='bold', ha='right')

    ax1.axhline(y=95, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax1.text(85, 97, '95%', fontsize=10, color='red', fontweight='bold', ha='right')

    # Add critical example (64 GPUs, 61 days)
    critical_day = 61
    critical_gpus = 64
    lambda_critical = critical_gpus / mtbf
    prob_critical = (1 - np.exp(-lambda_critical * critical_day)) * 100

    ax1.plot(critical_day, prob_critical, 'r*', markersize=20,
            markeredgecolor='black', markeredgewidth=2)
    ax1.annotate(f'GPT-3 Training\n64 GPUs × 61 days\n≈ 95% failure risk',
                xy=(critical_day, prob_critical), xytext=(30, 70),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    # Add formula
    ax1.text(0.05, 0.95, 'Exponential Reliability Model:\n' +
             'P(failure) = 1 - e^(-λt)\n' +
             'λ = N_GPUs / MTBF',
             transform=ax1.transAxes, ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3),
             fontsize=9, family='monospace')

    # Panel 2: Checkpointing Strategy Trade-off
    ax2 = axes[1]

    # Checkpoint intervals (in hours)
    checkpoint_intervals = np.array([0.5, 1, 2, 4, 6, 12, 24])

    # Storage overhead (% of training time)
    # More frequent checkpoints = more overhead
    storage_overhead = 100 * (1 / checkpoint_intervals) * 5  # 5 min per checkpoint
    storage_overhead = np.clip(storage_overhead, 0, 15)

    # Expected recovery time (average time lost)
    # = checkpoint_interval / 2 (on average, fail halfway through interval)
    recovery_time = checkpoint_intervals / 2

    # Create dual-axis plot
    ax2_twin = ax2.twinx()

    # Plot storage overhead
    line1 = ax2.plot(checkpoint_intervals, storage_overhead,
                    marker='s', markersize=10, linewidth=2.5,
                    color='#e74c3c', label='Storage Overhead', alpha=0.85)

    # Plot recovery time
    line2 = ax2_twin.plot(checkpoint_intervals, recovery_time,
                         marker='o', markersize=10, linewidth=2.5,
                         color='#3498db', label='Avg Recovery Time', alpha=0.85)

    ax2.set_xlabel('Checkpoint Interval (Hours)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Storage Overhead (% of Training Time)',
                   fontweight='bold', fontsize=12, color='#e74c3c')
    ax2_twin.set_ylabel('Avg Recovery Time (Hours)',
                       fontweight='bold', fontsize=12, color='#3498db')
    ax2.set_title('Checkpointing Trade-off: Overhead vs Recovery Time',
                  fontweight='bold', fontsize=13)

    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2_twin.tick_params(axis='y', labelcolor='#3498db')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Mark optimal point (2-4 hours)
    optimal_idx = 2  # 2 hours
    ax2.plot(checkpoint_intervals[optimal_idx], storage_overhead[optimal_idx],
            'g*', markersize=25, markeredgecolor='black', markeredgewidth=2)

    ax2.annotate('Optimal Balance\n2-4 hour intervals',
                xy=(checkpoint_intervals[optimal_idx], storage_overhead[optimal_idx]),
                xytext=(8, 5),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, frameon=True, fancybox=True,
              shadow=True, fontsize=10, loc='upper right')

    # Add best practices
    ax2.text(0.05, 0.05, 'Best Practices:\n' +
             '✓ Every 1-2 hours (< 5% overhead)\n' +
             '✓ Keep last 3 checkpoints\n' +
             '✓ Async save (non-blocking)\n' +
             '✓ Verify checkpoint integrity',
             transform=ax2.transAxes, ha='left', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
             fontsize=8)

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/failure_checkpointing.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    create_failure_checkpointing()
    print(f"\n✓ Section 14 Visualization 3/8 completed!")
    print(f"✓ Output: {OUTPUT_DIR}/failure_checkpointing.png")
