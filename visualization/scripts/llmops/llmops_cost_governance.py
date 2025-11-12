"""
Visualization 8 of 8 for Section 14: Cost Optimization & Governance

This script visualizes:
- Token usage optimization strategies (caching, batching, quantization)
- Governance audit trail and compliance framework

Output: visualization/images/llmops/cost_governance.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
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
print("SECTION 14 - VISUALIZATION 8/8: Cost Optimization & Governance")
print("=" * 80)

def create_cost_governance():
    """Two-panel visualization: cost optimization and governance framework"""
    fig = plt.figure(figsize=(18, 9))

    # Panel 1: Cost Optimization Strategies
    ax1 = fig.add_subplot(1, 2, 1)

    # Optimization techniques and their impact
    techniques = ['No\nOptimization', 'Caching\n(Responses)', 'Batching\n(Requests)',
                 'Quantization\n(INT8)', 'All\nCombined']

    # Cost savings (% reduction)
    cost_reduction = [0, 35, 20, 55, 75]

    # Latency impact (ms)
    latency_baseline = 250  # ms
    latency_values = [250, 245, 280, 240, 265]  # Batching adds latency

    # Implementation complexity (1-5 scale)
    complexity = [1, 2, 3, 4, 5]

    # Create stacked visualization
    x = np.arange(len(techniques))

    # Cost bars
    bars = ax1.bar(x, cost_reduction, color=['#e74c3c', '#27ae60', '#3498db',
                                             '#9b59b6', '#f39c12'],
                   alpha=0.85, edgecolor='black', linewidth=1.5)

    # Add percentage labels
    for i, (bar, reduction, latency, comp) in enumerate(zip(bars, cost_reduction,
                                                            latency_values, complexity)):
        height = bar.get_height()

        # Cost reduction label
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{reduction}%\nsavings',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Latency impact
        latency_diff = latency - latency_baseline
        latency_symbol = '↑' if latency_diff > 0 else '↓' if latency_diff < 0 else '→'
        latency_color = 'red' if latency_diff > 0 else 'green' if latency_diff < 0 else 'gray'

        ax1.text(bar.get_x() + bar.get_width()/2., height - 10,
                f'Latency:\n{latency}ms {latency_symbol}',
                ha='center', va='top', fontsize=8, color=latency_color)

        # Complexity stars
        stars = '★' * comp + '☆' * (5 - comp)
        ax1.text(bar.get_x() + bar.get_width()/2., -8,
                stars,
                ha='center', va='top', fontsize=8, color='orange')

    ax1.set_ylabel('Cost Reduction (%)', fontweight='bold', fontsize=12)
    ax1.set_title('Cost Optimization Strategies: Impact vs Complexity\n(★ = Implementation Difficulty)',
                  fontweight='bold', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(techniques, fontsize=10)
    ax1.set_ylim(-15, 85)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='black', linewidth=1)

    # Add ROI ranking
    ax1.text(0.98, 0.95, 'Implementation Priority:\n' +
             '1. Caching (35% save, low complexity)\n' +
             '2. Quantization (55% save, medium)\n' +
             '3. Batching (20% save, adds latency)\n' +
             '4. Combined (75% save, high complexity)',
             transform=ax1.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.4),
             fontsize=9)

    # Add cost breakdown example
    ax1.text(0.02, 0.05, 'Monthly Cost Example:\n' +
             'Baseline: $50,000\n' +
             'After optimization: $12,500\n' +
             'Annual savings: $450,000',
             transform=ax1.transAxes, ha='left', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.4),
             fontsize=9, fontweight='bold')

    # Panel 2: Governance & Audit Trail
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Governance Framework: Audit Trail & Compliance',
                  fontweight='bold', fontsize=14, pad=20)

    # Governance layers (pyramid structure)
    layers = [
        {
            'name': 'COMPLIANCE & REGULATIONS',
            'items': ['GDPR', 'HIPAA', 'SOC 2', 'Industry Standards'],
            'y': 8.5,
            'width': 8,
            'color': '#e74c3c'
        },
        {
            'name': 'AUDIT LOGGING',
            'items': ['Model versions', 'Data lineage', 'Access logs', 'Change history'],
            'y': 6.5,
            'width': 7,
            'color': '#e67e22'
        },
        {
            'name': 'ACCESS CONTROL',
            'items': ['RBAC', 'API keys', 'Rate limiting', 'IP whitelisting'],
            'y': 4.8,
            'width': 6,
            'color': '#f39c12'
        },
        {
            'name': 'MONITORING & ALERTING',
            'items': ['Drift detection', 'Bias monitoring', 'Performance metrics', 'Cost tracking'],
            'y': 3.1,
            'width': 5,
            'color': '#27ae60'
        },
        {
            'name': 'DOCUMENTATION',
            'items': ['Model cards', 'API docs', 'Runbooks', 'Incident reports'],
            'y': 1.5,
            'width': 4,
            'color': '#3498db'
        }
    ]

    for layer in layers:
        # Layer box
        x_start = (10 - layer['width']) / 2
        box = FancyBboxPatch((x_start, layer['y'] - 0.4), layer['width'], 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor=layer['color'],
                            edgecolor='black',
                            linewidth=2.5, alpha=0.85)
        ax2.add_patch(box)

        # Layer name
        ax2.text(5, layer['y'] + 0.15, layer['name'],
                ha='center', va='center',
                fontweight='bold', fontsize=11, color='white')

        # Items
        items_text = ' | '.join(layer['items'])
        ax2.text(5, layer['y'] - 0.15, items_text,
                ha='center', va='center',
                fontsize=8, color='white')

    # Add audit trail example
    audit_box = FancyBboxPatch((0.5, 0.2), 9, 0.7,
                              boxstyle="round,pad=0.1",
                              facecolor='#ecf0f1',
                              edgecolor='black',
                              linewidth=2, alpha=0.9)
    ax2.add_patch(audit_box)

    ax2.text(5, 0.65, 'Example Audit Entry:',
            ha='center', va='center',
            fontweight='bold', fontsize=10, color='#2c3e50')
    ax2.text(5, 0.35, '2024-11-13 14:23:45 | user@company.com | DEPLOY | model-v2.3.1 | ' +
             'staging→production | approval: manager@company.com',
            ha='center', va='center',
            fontsize=7, color='#2c3e50', family='monospace')

    # Add compliance checklist
    checklist = [
        ('✓', 'All model versions tracked in registry', 'green'),
        ('✓', 'Data lineage documented end-to-end', 'green'),
        ('✓', 'Access logs retained for 7 years', 'green'),
        ('⚠', 'Bias audit pending (scheduled Q4)', 'orange'),
        ('✗', 'GDPR right-to-explanation not implemented', 'red')
    ]

    checklist_y = 9.5
    for symbol, text, color in checklist:
        ax2.text(0.3, checklist_y, symbol, ha='left', va='top',
                fontsize=14, fontweight='bold', color=color)
        ax2.text(0.8, checklist_y, text, ha='left', va='top',
                fontsize=8, color='#2c3e50')
        checklist_y -= 0.35

    # Add governance score
    score_box = FancyBboxPatch((7, 9.2), 2.5, 0.8,
                              boxstyle="round,pad=0.15",
                              facecolor='#27ae60',
                              edgecolor='black',
                              linewidth=2.5, alpha=0.9)
    ax2.add_patch(score_box)
    ax2.text(8.25, 9.75, 'Compliance', ha='center', va='center',
            fontweight='bold', fontsize=10, color='white')
    ax2.text(8.25, 9.45, 'Score: 85%', ha='center', va='center',
            fontweight='bold', fontsize=12, color='white')

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/cost_governance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    create_cost_governance()
    print(f"\n✓ Section 14 Visualization 8/8 completed!")
    print(f"✓ Output: {OUTPUT_DIR}/cost_governance.png")
    print(f"\n{'='*80}")
    print("ALL SECTION 14 VISUALIZATIONS COMPLETED!")
    print(f"{'='*80}")
