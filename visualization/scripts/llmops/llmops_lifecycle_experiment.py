"""
Visualization 1 of 8 for Section 14: LLMOps Lifecycle & Experiment Tracking

This script visualizes:
- Model lifecycle phases (Dev → Train → Deploy → Monitor → Retrain)
- Experiment tracking components (code, data, config, metrics, artifacts)

Output: visualization/images/llmops/lifecycle_experiment_tracking.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
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
print("SECTION 14 - VISUALIZATION 1/8: Model Lifecycle & Experiment Tracking")
print("=" * 80)

def create_lifecycle_experiment():
    """Two-panel visualization: lifecycle and experiment tracking"""
    fig = plt.figure(figsize=(18, 9))

    # Panel 1: Model Lifecycle Phases
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('LLMOps Lifecycle: Continuous Improvement Loop',
                  fontweight='bold', fontsize=14, pad=20)

    # Define lifecycle phases in a circle
    phases = [
        ('Development', 'Define requirements\nPrepare datasets', '#3498db'),
        ('Training', 'Pre-train/Fine-tune\nHyperparameter tuning', '#e67e22'),
        ('Evaluation', 'Accuracy metrics\nBenchmark testing', '#9b59b6'),
        ('Deployment', 'Staging → Production\nA/B testing', '#27ae60'),
        ('Monitoring', 'Performance metrics\nDrift detection', '#e74c3c'),
        ('Feedback', 'User feedback\nError analysis', '#f39c12'),
        ('Retrain', 'New data ingestion\nModel update', '#1abc9c')
    ]

    center_x, center_y = 5, 5
    radius = 3.5
    n_phases = len(phases)

    # Draw circular lifecycle
    for i, (phase, desc, color) in enumerate(phases):
        angle = 2 * np.pi * i / n_phases - np.pi/2  # Start at top

        # Phase box position
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)

        # Draw phase box
        box = FancyBboxPatch((x - 0.8, y - 0.45), 1.6, 0.9,
                            boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black',
                            linewidth=2.5, alpha=0.85)
        ax1.add_patch(box)

        # Phase name
        ax1.text(x, y + 0.15, phase, ha='center', va='center',
                fontweight='bold', fontsize=10, color='white')

        # Description
        ax1.text(x, y - 0.15, desc, ha='center', va='center',
                fontsize=7, color='white', style='italic')

        # Arrow to next phase
        next_angle = 2 * np.pi * (i + 1) / n_phases - np.pi/2
        arrow_start_x = center_x + (radius - 0.3) * np.cos(angle + 2*np.pi/(2*n_phases))
        arrow_start_y = center_y + (radius - 0.3) * np.sin(angle + 2*np.pi/(2*n_phases))
        arrow_end_x = center_x + (radius - 0.3) * np.cos(next_angle - 2*np.pi/(2*n_phases))
        arrow_end_y = center_y + (radius - 0.3) * np.sin(next_angle - 2*np.pi/(2*n_phases))

        arrow = FancyArrowPatch((arrow_start_x, arrow_start_y),
                               (arrow_end_x, arrow_end_y),
                               arrowstyle='->', mutation_scale=25,
                               linewidth=3, color='black', alpha=0.6,
                               connectionstyle="arc3,rad=0.2")
        ax1.add_patch(arrow)

    # Central "Continuous" label
    circle = Circle((center_x, center_y), 1.2, facecolor='#ecf0f1',
                   edgecolor='black', linewidth=2.5, alpha=0.9)
    ax1.add_patch(circle)
    ax1.text(center_x, center_y + 0.2, 'CONTINUOUS', ha='center', va='center',
            fontweight='bold', fontsize=11, color='#2c3e50')
    ax1.text(center_x, center_y - 0.2, 'LOOP', ha='center', va='center',
            fontweight='bold', fontsize=11, color='#2c3e50')

    # Add annotation
    ax1.text(5, 0.5, 'Average cycle: 3-6 months | Critical: Monitor → Retrain trigger',
            ha='center', va='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

    # Panel 2: Experiment Tracking Components
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Experiment Tracking: 5 Essential Components',
                  fontweight='bold', fontsize=14, pad=20)

    # Experiment tracking components
    components = [
        {
            'name': 'CODE',
            'desc': '• Git commit hash\n• Script versions\n• Library versions',
            'color': '#3498db',
            'y': 8.5
        },
        {
            'name': 'DATA',
            'desc': '• Dataset version/hash\n• Train/val/test splits\n• Preprocessing config',
            'color': '#e67e22',
            'y': 7.0
        },
        {
            'name': 'CONFIG',
            'desc': '• Hyperparameters\n• Model architecture\n• Training settings',
            'color': '#9b59b6',
            'y': 5.5
        },
        {
            'name': 'METRICS',
            'desc': '• Loss curves\n• Validation accuracy\n• Training time',
            'color': '#27ae60',
            'y': 4.0
        },
        {
            'name': 'ARTIFACTS',
            'desc': '• Model checkpoints\n• Logs & tensorboard\n• Generated samples',
            'color': '#e74c3c',
            'y': 2.5
        }
    ]

    for i, comp in enumerate(components):
        # Component box
        box = FancyBboxPatch((1, comp['y'] - 0.5), 8, 1.2,
                            boxstyle="round,pad=0.15",
                            facecolor=comp['color'],
                            edgecolor='black',
                            linewidth=2.5, alpha=0.85)
        ax2.add_patch(box)

        # Component name
        ax2.text(2, comp['y'] + 0.25, comp['name'],
                ha='left', va='center',
                fontweight='bold', fontsize=12, color='white')

        # Description
        ax2.text(2, comp['y'] - 0.15, comp['desc'],
                ha='left', va='top',
                fontsize=8, color='white')

        # Checkmark icon
        ax2.text(8.5, comp['y'] + 0.1, '✓',
                ha='center', va='center',
                fontweight='bold', fontsize=20, color='white')

    # Reproducibility badge
    badge = FancyBboxPatch((2.5, 0.5), 5, 1,
                          boxstyle="round,pad=0.2",
                          facecolor='#f39c12',
                          edgecolor='black',
                          linewidth=3, alpha=0.9)
    ax2.add_patch(badge)
    ax2.text(5, 1.0, '100% REPRODUCIBLE EXPERIMENT',
            ha='center', va='center',
            fontweight='bold', fontsize=11, color='white')
    ax2.text(5, 0.65, 'Track all 5 components for every training run',
            ha='center', va='center',
            fontsize=8, color='white', style='italic')

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/lifecycle_experiment_tracking.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    create_lifecycle_experiment()
    print(f"\n✓ Section 14 Visualization 1/8 completed!")
    print(f"✓ Output: {OUTPUT_DIR}/lifecycle_experiment_tracking.png")
