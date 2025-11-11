"""
Section 11: Safety, Ethics, and Bias Mitigation - Safety Monitoring Dashboard Visualizations
Creates 2 visualizations for continuous safety monitoring and alert systems
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

def create_safety_monitoring_visualization():
    """
    Create 2-panel visualization for safety monitoring:
    1. Time series of safety metrics during deployment
    2. Alert system effectiveness and response tracking
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Continuous Safety Monitoring: Real-time Metrics and Alert Management', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Panel 1: Safety Metrics Over Time
    ax1 = axes[0]
    
    # Simulate 30 days of deployment
    days = np.arange(1, 31)
    
    # Toxicity rate (percentage of responses flagged as toxic)
    np.random.seed(42)
    toxicity_rate = 3.5 + 2.0 * np.sin(days * 0.3) + np.random.normal(0, 0.5, len(days))
    toxicity_rate = np.clip(toxicity_rate, 1.0, 8.0)
    
    # Bias metric (WEAT score magnitude - lower is better)
    bias_metric = 0.45 + 0.15 * np.sin(days * 0.25) + np.random.normal(0, 0.03, len(days))
    bias_metric = np.clip(bias_metric, 0.2, 0.7)
    
    # Hallucination rate (percentage of responses with factual errors)
    hallucination_rate = 2.8 + 1.5 * np.sin(days * 0.35) + np.random.normal(0, 0.4, len(days))
    hallucination_rate = np.clip(hallucination_rate, 1.0, 6.0)
    
    # Create twin axes for different scales
    ax1_twin1 = ax1.twinx()
    ax1_twin2 = ax1.twinx()
    
    # Offset the right spine of ax1_twin2
    ax1_twin2.spines['right'].set_position(('outward', 60))
    
    # Plot metrics
    line1 = ax1.plot(days, toxicity_rate, 'o-', color='#E74C3C', linewidth=2.5,
                    markersize=6, markeredgecolor='darkred', markeredgewidth=1,
                    label='Toxicity Rate (%)', alpha=0.8)
    line2 = ax1_twin1.plot(days, bias_metric, 's-', color='#3498DB', linewidth=2.5,
                          markersize=6, markeredgecolor='darkblue', markeredgewidth=1,
                          label='Bias Score (WEAT)', alpha=0.8)
    line3 = ax1_twin2.plot(days, hallucination_rate, '^-', color='#2ECC71', linewidth=2.5,
                          markersize=6, markeredgecolor='darkgreen', markeredgewidth=1,
                          label='Hallucination Rate (%)', alpha=0.8)
    
    # Add threshold lines
    ax1.axhline(y=5.0, color='red', linestyle='--', linewidth=2, alpha=0.4, label='Toxicity Threshold')
    ax1_twin1.axhline(y=0.5, color='blue', linestyle='--', linewidth=2, alpha=0.4, label='Bias Threshold')
    ax1_twin2.axhline(y=4.0, color='green', linestyle='--', linewidth=2, alpha=0.4, label='Hallucination Threshold')
    
    # Highlight violation periods
    violations = toxicity_rate > 5.0
    if np.any(violations):
        violation_periods = days[violations]
        for vday in violation_periods:
            ax1.axvspan(vday - 0.5, vday + 0.5, alpha=0.15, color='red')
    
    # Labels
    ax1.set_xlabel('Days Since Deployment', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Toxicity Rate (%)', fontsize=10, fontweight='bold', color='#E74C3C')
    ax1_twin1.set_ylabel('Bias Score (WEAT)', fontsize=10, fontweight='bold', color='#3498DB')
    ax1_twin2.set_ylabel('Hallucination Rate (%)', fontsize=10, fontweight='bold', color='#2ECC71')
    
    ax1.set_title('(1) Safety Metrics Timeline\n30-Day Deployment Monitoring', 
                  fontsize=12, fontweight='bold', pad=10)
    
    # Color y-axis ticks to match lines
    ax1.tick_params(axis='y', labelcolor='#E74C3C')
    ax1_twin1.tick_params(axis='y', labelcolor='#3498DB')
    ax1_twin2.tick_params(axis='y', labelcolor='#2ECC71')
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=9, framealpha=0.9)
    
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 31])
    
    # Panel 2: Alert System Dashboard
    ax2 = axes[1]
    
    # Alert categories and their counts
    alert_categories = ['Toxicity\nSpike', 'Bias\nDrift', 'Hallucination\nIncrease', 
                       'User\nReports', 'Policy\nViolations']
    
    # Alerts triggered, acknowledged, resolved
    triggered = [12, 8, 15, 22, 5]
    acknowledged = [12, 8, 14, 20, 5]
    resolved = [10, 7, 12, 18, 4]
    
    x = np.arange(len(alert_categories))
    width = 0.25
    
    # Create grouped bars
    bars1 = ax2.bar(x - width, triggered, width, label='Triggered',
                   color='#E74C3C', alpha=0.9, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x, acknowledged, width, label='Acknowledged',
                   color='#F39C12', alpha=0.9, edgecolor='black', linewidth=1.5)
    bars3 = ax2.bar(x + width, resolved, width, label='Resolved',
                   color='#27AE60', alpha=0.9, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Calculate resolution rates
    resolution_rates = [(resolved[i] / triggered[i] * 100) if triggered[i] > 0 else 0 
                       for i in range(len(alert_categories))]
    
    # Add resolution rate annotations
    for i, rate in enumerate(resolution_rates):
        ax2.text(i, triggered[i] + 2.5, f'{rate:.0f}%\nresolved',
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.6))
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(alert_categories, fontsize=10)
    ax2.set_ylabel('Number of Alerts', fontsize=11, fontweight='bold')
    ax2.set_title('(2) Alert System Performance\n30-Day Alert Tracking and Resolution', 
                  fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 27])
    
    # Add summary statistics box
    total_triggered = sum(triggered)
    total_resolved = sum(resolved)
    overall_resolution = (total_resolved / total_triggered * 100) if total_triggered > 0 else 0
    avg_response_time = 2.4  # hours
    
    textstr = f'Overall Statistics:\n'
    textstr += f'• Total Alerts: {total_triggered}\n'
    textstr += f'• Resolution Rate: {overall_resolution:.1f}%\n'
    textstr += f'• Avg Response: {avg_response_time}h'
    
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='black', linewidth=2)
    ax2.text(0.02, 0.97, textstr, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left', bbox=props,
            fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/safety_monitoring_dashboard.png", dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/safety_monitoring_dashboard.png")
    plt.close()

if __name__ == "__main__":
    print("Generating Section 11 visualizations: Safety Monitoring Dashboard...")
    create_safety_monitoring_visualization()
    print("\n✓ All safety monitoring visualizations generated successfully!")
    print(f"✓ Output directory: {OUTPUT_DIR}")
