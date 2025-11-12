"""
Visualization 5 of 8 for Section 14: Monitoring Dashboard & System Health

This script visualizes:
- Real-time monitoring metrics (accuracy, latency, cost, drift)
- System health composite scoring

Output: visualization/images/llmops/monitoring_dashboard.png
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
print("SECTION 14 - VISUALIZATION 5/8: Monitoring Dashboard & System Health")
print("=" * 80)

def create_monitoring_dashboard():
    """Two-panel visualization: monitoring metrics and system health scoring"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Production Monitoring Dashboard: Real-Time Metrics & Health Scoring',
                 fontsize=16, fontweight='bold', y=0.995)

    # Panel 1: Accuracy Over Time
    ax1 = axes[0, 0]

    # Simulate 30 days of accuracy monitoring
    days = np.arange(1, 31)
    baseline_accuracy = 85.0

    # Accuracy degrades gradually
    noise = np.random.normal(0, 0.5, len(days))
    drift = -0.15 * days  # Linear degradation
    accuracy = baseline_accuracy + drift + noise

    # Plot accuracy
    ax1.plot(days, accuracy, linewidth=2.5, color='#3498db',
            marker='o', markersize=6, label='Current Accuracy', alpha=0.85)

    # Baseline
    ax1.axhline(y=baseline_accuracy, color='green', linestyle='--',
               linewidth=2, label='Deployment Baseline (85%)', alpha=0.7)

    # Warning threshold (-5%)
    ax1.axhline(y=baseline_accuracy * 0.95, color='orange', linestyle='--',
               linewidth=2, label='Warning Threshold (-5%)', alpha=0.7)

    # Critical threshold (-10%)
    ax1.axhline(y=baseline_accuracy * 0.90, color='red', linestyle='--',
               linewidth=2, label='Critical Threshold (-10%)', alpha=0.7)

    # Highlight degradation point
    degrade_day = 23
    ax1.plot(degrade_day, accuracy[degrade_day-1], 'r*',
            markersize=20, markeredgecolor='black', markeredgewidth=2)
    ax1.annotate(f'ALERT!\nAccuracy: {accuracy[degrade_day-1]:.1f}%\nΔ = -4.2%',
                xy=(degrade_day, accuracy[degrade_day-1]), xytext=(15, 78),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    ax1.set_xlabel('Days Since Deployment', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=11)
    ax1.set_title('Metric 1: Accuracy Monitoring (Daily Evaluation)',
                  fontweight='bold', fontsize=12)
    ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(76, 87)

    # Panel 2: Latency Distribution
    ax2 = axes[0, 1]

    # Simulate latency distribution (p50, p90, p95, p99)
    hours = np.arange(0, 24)

    p50_latency = 80 + 20 * np.sin(hours * np.pi / 12) + np.random.normal(0, 3, len(hours))
    p90_latency = 150 + 30 * np.sin(hours * np.pi / 12) + np.random.normal(0, 5, len(hours))
    p99_latency = 300 + 50 * np.sin(hours * np.pi / 12) + np.random.normal(0, 10, len(hours))

    ax2.fill_between(hours, p50_latency, alpha=0.3, color='#27ae60', label='p50 (Median)')
    ax2.fill_between(hours, p50_latency, p90_latency, alpha=0.3, color='#f39c12', label='p50-p90')
    ax2.fill_between(hours, p90_latency, p99_latency, alpha=0.3, color='#e74c3c', label='p90-p99')

    ax2.plot(hours, p50_latency, linewidth=2, color='#27ae60', alpha=0.8)
    ax2.plot(hours, p90_latency, linewidth=2, color='#f39c12', alpha=0.8)
    ax2.plot(hours, p99_latency, linewidth=2, color='#e74c3c', alpha=0.8)

    # SLA threshold
    ax2.axhline(y=500, color='red', linestyle='--', linewidth=2.5,
               label='SLA Threshold (500ms)', alpha=0.7)

    ax2.set_xlabel('Hour of Day (UTC)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Latency (ms)', fontweight='bold', fontsize=11)
    ax2.set_title('Metric 2: Latency Distribution (24-Hour Profile)',
                  fontweight='bold', fontsize=12)
    ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 23)
    ax2.set_xticks([0, 6, 12, 18, 23])

    # Panel 3: Cost Tracking
    ax3 = axes[1, 0]

    # Daily cost breakdown
    cost_categories = ['Compute\n(GPU)', 'Storage\n(Models)', 'API Calls\n(Inference)',
                      'Monitoring\n(Logs)', 'Network\n(Transfer)']
    daily_costs = [3200, 450, 1800, 250, 180]  # USD per day
    colors = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#27ae60']

    bars = ax3.bar(cost_categories, daily_costs, color=colors,
                   alpha=0.85, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, cost) in enumerate(zip(bars, daily_costs)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 100,
                f'${cost:,}\n/day',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Budget line
    budget = 6500
    ax3.axhline(y=budget, color='red', linestyle='--', linewidth=2.5,
               label=f'Daily Budget: ${budget:,}', alpha=0.7)

    # Total line
    total = sum(daily_costs)
    ax3.axhline(y=total, color='green', linestyle='-', linewidth=2.5,
               label=f'Current Total: ${total:,}', alpha=0.7)

    ax3.set_ylabel('Cost (USD per Day)', fontweight='bold', fontsize=11)
    ax3.set_title('Metric 3: Cost Breakdown (Daily Operating Expenses)',
                  fontweight='bold', fontsize=12)
    ax3.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_ylim(0, 7000)

    # Add utilization note
    utilization = (total / budget) * 100
    ax3.text(0.5, 0.95, f'Budget Utilization: {utilization:.1f}%\n({budget-total:,.0f} USD headroom)',
            transform=ax3.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.5),
            fontsize=9, fontweight='bold')

    # Panel 4: System Health Composite Score
    ax4 = axes[1, 1]

    # Health components and their scores
    components = ['Accuracy', 'Latency', 'Cost', 'Drift']
    scores = [0.75, 0.85, 0.90, 0.65]  # Normalized [0, 1]
    weights = [0.5, 0.2, 0.1, 0.2]  # Business priorities

    # Weighted health score
    health_score = sum(s * w for s, w in zip(scores, weights))

    x = np.arange(len(components))
    bars = ax4.barh(x, scores, color=['#3498db', '#27ae60', '#f39c12', '#e74c3c'],
                    alpha=0.85, edgecolor='black', linewidth=1.5)

    # Add score labels
    for i, (bar, score, weight) in enumerate(zip(bars, scores, weights)):
        width = bar.get_width()
        ax4.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{score*100:.0f}% (w={weight})',
                ha='left', va='center', fontsize=10, fontweight='bold')

    # Threshold lines
    ax4.axvline(x=0.7, color='orange', linestyle='--', linewidth=2,
               label='Warning (70%)', alpha=0.6)
    ax4.axvline(x=0.5, color='red', linestyle='--', linewidth=2,
               label='Critical (50%)', alpha=0.6)

    ax4.set_yticks(x)
    ax4.set_yticklabels(components, fontsize=11, fontweight='bold')
    ax4.set_xlabel('Normalized Score', fontweight='bold', fontsize=11)
    ax4.set_title('Metric 4: System Health Composite Scoring',
                  fontweight='bold', fontsize=12)
    ax4.set_xlim(0, 1.0)
    ax4.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax4.grid(axis='x', alpha=0.3, linestyle='--')

    # Overall health score
    health_color = '#27ae60' if health_score >= 0.7 else '#f39c12' if health_score >= 0.5 else '#e74c3c'
    health_status = 'HEALTHY' if health_score >= 0.7 else 'WARNING' if health_score >= 0.5 else 'CRITICAL'

    ax4.text(0.5, -0.15, f'Overall System Health: {health_score*100:.1f}%\nStatus: {health_status}',
            transform=ax4.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=health_color, alpha=0.6,
                     edgecolor='black', linewidth=2),
            fontsize=12, fontweight='bold', color='white')

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/monitoring_dashboard.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    create_monitoring_dashboard()
    print(f"\n✓ Section 14 Visualization 5/8 completed!")
    print(f"✓ Output: {OUTPUT_DIR}/monitoring_dashboard.png")
