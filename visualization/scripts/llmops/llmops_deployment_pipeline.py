"""
Visualization 4 of 8 for Section 14: Deployment Pipeline & CI/CD Workflow

This script visualizes:
- CI/CD deployment stages (Dev → Staging → Production)
- Deployment decision criteria (accuracy, safety, efficiency, cost)

Output: visualization/images/llmops/deployment_pipeline.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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
print("SECTION 14 - VISUALIZATION 4/8: Deployment Pipeline & CI/CD")
print("=" * 80)

def create_deployment_pipeline():
    """Two-panel visualization: CI/CD stages and deployment criteria"""
    fig = plt.figure(figsize=(18, 9))

    # Panel 1: CI/CD Deployment Pipeline
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('CI/CD Deployment Pipeline: Gate-Based Progression',
                  fontweight='bold', fontsize=14, pad=20)

    # Define deployment stages
    stages = [
        {
            'name': 'DEVELOPMENT',
            'tests': ['Unit Tests', 'Integration Tests', 'Code Review'],
            'gate': 'All tests pass\nCode coverage > 80%',
            'y': 8.5,
            'color': '#3498db'
        },
        {
            'name': 'STAGING',
            'tests': ['Accuracy Tests', 'Safety Tests', 'Performance Tests'],
            'gate': 'Accuracy ≥ baseline\nNo safety violations',
            'y': 5.5,
            'color': '#f39c12'
        },
        {
            'name': 'PRODUCTION',
            'tests': ['A/B Testing', 'Canary Deploy', 'Full Rollout'],
            'gate': 'User metrics stable\nNo regressions',
            'y': 2.5,
            'color': '#27ae60'
        }
    ]

    for i, stage in enumerate(stages):
        # Stage box
        box = FancyBboxPatch((1, stage['y'] - 0.6), 3.5, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor=stage['color'],
                            edgecolor='black',
                            linewidth=2.5, alpha=0.85)
        ax1.add_patch(box)

        # Stage name
        ax1.text(2.75, stage['y'] + 0.35, stage['name'],
                ha='center', va='center',
                fontweight='bold', fontsize=12, color='white')

        # Tests list
        tests_text = '\n'.join(['• ' + t for t in stage['tests']])
        ax1.text(2.75, stage['y'] - 0.1, tests_text,
                ha='center', va='center',
                fontsize=8, color='white')

        # Gate criteria box
        gate_box = FancyBboxPatch((5.5, stage['y'] - 0.4), 3.5, 0.8,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#ecf0f1',
                                 edgecolor='black',
                                 linewidth=2, alpha=0.9)
        ax1.add_patch(gate_box)

        # Gate icon
        ax1.text(5.8, stage['y'], '🚪', ha='center', va='center', fontsize=16)

        # Gate text
        ax1.text(7.25, stage['y'], stage['gate'],
                ha='center', va='center',
                fontsize=8, fontweight='bold', color='#2c3e50')

        # Arrow to next stage
        if i < len(stages) - 1:
            arrow = FancyArrowPatch((2.75, stage['y'] - 0.7),
                                   (2.75, stages[i+1]['y'] + 0.7),
                                   arrowstyle='->', mutation_scale=30,
                                   linewidth=4, color='black', alpha=0.6)
            ax1.add_patch(arrow)

            # "PASS" label
            ax1.text(3.5, (stage['y'] + stages[i+1]['y']) / 2, 'PASS ✓',
                    ha='left', va='center',
                    fontsize=10, fontweight='bold', color='green',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))

    # Rollback arrow
    rollback = FancyArrowPatch((1.5, 2.0), (1.5, 8.0),
                              arrowstyle='<->', mutation_scale=25,
                              linewidth=3, color='red',
                              linestyle='--', alpha=0.6)
    ax1.add_patch(rollback)
    ax1.text(0.5, 5, 'ROLLBACK\nIF FAIL', ha='center', va='center',
            fontsize=9, fontweight='bold', color='red', rotation=90,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe6e6', alpha=0.7))

    # Timeline annotation
    ax1.text(5, 0.7, 'Average Pipeline Duration: 2-4 hours (automated) | Manual approval gates optional',
            ha='center', va='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.3))

    # Panel 2: Deployment Decision Criteria
    ax2 = fig.add_subplot(1, 2, 2)

    # Multi-criteria decision radar chart
    categories = ['Accuracy', 'Safety', 'Efficiency', 'Cost']
    n_cats = len(categories)

    # Example scenarios
    scenarios = [
        {
            'name': 'Base Model',
            'values': [0.7, 0.6, 0.5, 0.8],
            'color': '#e74c3c'
        },
        {
            'name': 'Fine-tuned (Staging)',
            'values': [0.85, 0.75, 0.7, 0.7],
            'color': '#f39c12'
        },
        {
            'name': 'Production Ready',
            'values': [0.9, 0.95, 0.85, 0.75],
            'color': '#27ae60'
        }
    ]

    # Compute angle for each category
    angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
    angles += angles[:1]  # Complete the circle

    ax2 = plt.subplot(122, projection='polar')
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)

    # Plot each scenario
    for scenario in scenarios:
        values = scenario['values']
        values += values[:1]  # Complete the circle
        ax2.plot(angles, values, 'o-', linewidth=2.5,
                label=scenario['name'], color=scenario['color'], alpha=0.85)
        ax2.fill(angles, values, alpha=0.15, color=scenario['color'])

    # Set category labels
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=11, fontweight='bold')

    # Set radial limits
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)

    ax2.set_title('Deployment Decision: Multi-Criteria Assessment\n(All 4 criteria must be satisfied)',
                  fontweight='bold', fontsize=13, pad=20)

    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
              frameon=True, fancybox=True, shadow=True, fontsize=10)

    # Add threshold line
    threshold = [0.8] * (n_cats + 1)
    ax2.plot(angles, threshold, 'r--', linewidth=2.5,
            label='Minimum Threshold', alpha=0.7)

    # Add decision formula annotation
    ax2.text(0.5, -0.2, 'Decision = f(Accuracy, Safety, Efficiency, Cost)\n' +
             'Deploy if: ALL criteria ≥ 80% threshold',
             transform=ax2.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5),
             fontsize=9, fontweight='bold')

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/deployment_pipeline.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    create_deployment_pipeline()
    print(f"\n✓ Section 14 Visualization 4/8 completed!")
    print(f"✓ Output: {OUTPUT_DIR}/deployment_pipeline.png")
