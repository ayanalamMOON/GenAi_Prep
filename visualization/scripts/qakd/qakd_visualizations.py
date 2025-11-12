"""
QAKD (Quantization-Aware Knowledge Distillation) Visualization Script
Generates publication-quality visualizations for QAKD section of LLM Study Material
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Output directory (two levels up, then into images/qakd)
output_dir = Path(__file__).parent.parent.parent / 'images' / 'qakd'
output_dir.mkdir(parents=True, exist_ok=True)

def create_qakd_framework():
    """QAKD framework showing teacher-student distillation with quantization awareness"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('QAKD Framework: Knowledge Distillation + Quantization Awareness',
                 fontsize=14, weight='bold', y=1.02)

    # Left panel: Dual Loss Architecture
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Dual Loss Architecture', fontsize=12, weight='bold', pad=10)

    # Teacher model (top)
    teacher_box = FancyBboxPatch((1, 7.5), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#d4edda', edgecolor='#28a745', linewidth=2.5)
    ax1.add_patch(teacher_box)
    ax1.text(2.5, 8.7, 'Teacher Model', fontsize=11, weight='bold', ha='center')
    ax1.text(2.5, 8.2, '(Frozen, Full Precision)', fontsize=8, ha='center', style='italic')

    # Student model (bottom)
    student_box = FancyBboxPatch((1, 0.5), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#fff3cd', edgecolor='#f39c12', linewidth=2.5)
    ax1.add_patch(student_box)
    ax1.text(2.5, 1.7, 'Student Model', fontsize=11, weight='bold', ha='center')
    ax1.text(2.5, 1.2, '(Trainable + Fake Quant)', fontsize=8, ha='center', style='italic')
    ax1.text(2.5, 0.8, 'W_q = Δ·Round(W/Δ)', fontsize=8, ha='center', family='monospace')

    # Input
    input_circle = Circle((2.5, 4), 0.4, facecolor='#e8f4f8', edgecolor='#2E86AB', linewidth=2)
    ax1.add_patch(input_circle)
    ax1.text(2.5, 4, 'Input\nx', fontsize=9, ha='center', va='center')

    # Arrows from input
    ax1.annotate('', xy=(2.5, 7.4), xytext=(2.5, 4.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='#28a745'))
    ax1.annotate('', xy=(2.5, 2.1), xytext=(2.5, 3.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='#f39c12'))

    # Loss 1: KD Loss
    kd_box = FancyBboxPatch((5, 6.5), 3.5, 2, boxstyle="round,pad=0.1",
                            facecolor='#ffe6e6', edgecolor='#dc3545', linewidth=2)
    ax1.add_patch(kd_box)
    ax1.text(6.75, 8, '𝓛_KD (Distillation)', fontsize=11, weight='bold', ha='center')
    ax1.text(6.75, 7.5, 'KL(p_teacher || p_student)', fontsize=9, ha='center', family='monospace')
    ax1.text(6.75, 7, 'with temperature τ=3', fontsize=8, ha='center', style='italic')

    # Loss 2: Quantization Loss
    quant_box = FancyBboxPatch((5, 0.5), 3.5, 2, boxstyle="round,pad=0.1",
                               facecolor='#e6f0ff', edgecolor='#007bff', linewidth=2)
    ax1.add_patch(quant_box)
    ax1.text(6.75, 2, '𝓛_quant (Noise Reg.)', fontsize=11, weight='bold', ha='center')
    ax1.text(6.75, 1.5, '||W_q - W||²_F', fontsize=9, ha='center', family='monospace')
    ax1.text(6.75, 1, 'Penalty: λ = 5×10⁻⁴', fontsize=8, ha='center', style='italic')

    # Arrows to losses
    ax1.annotate('', xy=(4.9, 7.5), xytext=(4.1, 8.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#28a745'))
    ax1.annotate('', xy=(4.9, 7.5), xytext=(4.1, 1.3),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#f39c12'))
    ax1.annotate('', xy=(4.9, 1.5), xytext=(4.1, 1.3),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#f39c12'))

    # Total loss
    total_box = FancyBboxPatch((5.5, 3.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                               facecolor='#f0e6ff', edgecolor='#6f42c1', linewidth=2.5)
    ax1.add_patch(total_box)
    ax1.text(6.75, 4.5, '𝓛_total', fontsize=11, weight='bold', ha='center')
    ax1.text(6.75, 4, '𝓛_KD + λ·𝓛_quant', fontsize=9, ha='center', family='monospace')

    ax1.annotate('', xy=(6.75, 3.5), xytext=(6.75, 6.4),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#dc3545'))
    ax1.annotate('', xy=(6.75, 3.5), xytext=(6.75, 2.6),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#007bff'))

    # Right panel: Training vs Inference
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Training vs Inference Quantization', fontsize=12, weight='bold', pad=10)

    # Training (top)
    ax2.text(5, 9, 'Training: Fake Quantization', fontsize=11, weight='bold', ha='center')

    train_flow = FancyBboxPatch((1, 6.5), 8, 2, boxstyle="round,pad=0.1",
                                facecolor='#fff8e1', edgecolor='#ff9800', linewidth=2)
    ax2.add_patch(train_flow)

    ax2.text(2, 8, 'W (FP32)', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    ax2.text(5, 8, 'W_q = Δ·Round(W/Δ)', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='#ffebee', edgecolor='red'))
    ax2.text(8, 8, 'Forward (W_q)', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='#e3f2fd', edgecolor='blue'))

    ax2.annotate('', xy=(3.5, 8), xytext=(2.7, 8),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax2.annotate('', xy=(6.8, 8), xytext=(6, 8),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax2.text(5, 7.2, 'Backward: Gradients flow to W (STE)', fontsize=9, ha='center', style='italic')
    ax2.text(5, 6.8, '∂𝓛/∂W ← ∂𝓛/∂W_q (approximation)', fontsize=8, ha='center', family='monospace')

    # Inference (bottom)
    ax2.text(5, 5.5, 'Inference: True Quantization', fontsize=11, weight='bold', ha='center')

    infer_flow = FancyBboxPatch((1, 2.5), 8, 2.5, boxstyle="round,pad=0.1",
                                facecolor='#e8f5e9', edgecolor='#4caf50', linewidth=2)
    ax2.add_patch(infer_flow)

    ax2.text(2.5, 4.3, 'W_trained\n(FP32)', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    ax2.text(5, 4.3, 'GPTQ/AWQ\nQuantize', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='#fff3cd', edgecolor='orange'))
    ax2.text(7.5, 4.3, 'W_int4\n(True INT4)', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='#d4edda', edgecolor='green'))

    ax2.annotate('', xy=(3.8, 4.3), xytext=(3.2, 4.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax2.annotate('', xy=(6.5, 4.3), xytext=(5.8, 4.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax2.text(5, 3.3, 'Memory: 7B → 1.75 GB', fontsize=9, ha='center', weight='bold', color='green')
    ax2.text(5, 2.9, 'Speed: 7× faster on CPU', fontsize=9, ha='center', weight='bold', color='green')

    # Key difference
    diff_box = FancyBboxPatch((1.5, 0.2), 7, 1.5, boxstyle="round,pad=0.1",
                              facecolor='#fce4ec', edgecolor='#e91e63', linewidth=2)
    ax2.add_patch(diff_box)
    ax2.text(5, 1.3, 'Key Difference', fontsize=10, weight='bold', ha='center')
    ax2.text(5, 0.9, 'Training: Simulate quantization noise (FP16 compute)', fontsize=8, ha='center')
    ax2.text(5, 0.5, 'Inference: Actual integer operations (INT4 compute)', fontsize=8, ha='center')

    plt.tight_layout()
    plt.savefig(output_dir / 'qakd_framework.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: qakd_framework.png")

def create_training_dynamics():
    """Show QAKD training dynamics over time"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('QAKD Training Dynamics: 7B→1.5B INT4 Compression',
                 fontsize=14, weight='bold', y=1.00)

    steps = np.linspace(0, 2500, 100)

    # Panel 1: Loss Curves
    ax1 = axes[0, 0]

    # KD Loss (decreases from 3.45 to 1.62)
    kd_loss = 3.45 * np.exp(-0.0008 * steps) + 1.2
    # Quantization Loss (decreases from 0.0028 to 0.0009)
    quant_loss = 0.0028 * np.exp(-0.0005 * steps) + 0.0008
    # Total Loss
    total_loss = kd_loss + 5e-4 * quant_loss * 1000  # Scale for visibility

    ax1.plot(steps, kd_loss, 'b-', linewidth=2, label='𝓛_KD (Distillation)', marker='o',
            markevery=10, markersize=4)
    ax1.plot(steps, quant_loss * 1000, 'r--', linewidth=2, label='𝓛_quant × 1000 (Scaled)',
            marker='s', markevery=10, markersize=4)

    ax1.set_xlabel('Training Step', fontsize=10, weight='bold')
    ax1.set_ylabel('Loss Value', fontsize=10, weight='bold')
    ax1.set_title('Loss Convergence', fontsize=11, weight='bold')
    ax1.legend(loc='upper right', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 2500)

    # Annotate key points
    ax1.annotate('Initial: High KD loss\n(student far from teacher)',
                xy=(0, 3.45), xytext=(300, 3.8),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax1.annotate('Converged: Teacher knowledge transferred',
                xy=(2500, 1.62), xytext=(1700, 2.2),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                fontsize=8, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Panel 2: Accuracy Gap
    ax2 = axes[0, 1]

    # Teacher accuracy (constant at 78.5%)
    teacher_acc = np.full_like(steps, 78.5)
    # Student accuracy (improves from 65% to 76.8%)
    student_acc = 65 + 11.8 * (1 - np.exp(-0.001 * steps))
    accuracy_gap = teacher_acc - student_acc

    ax2.plot(steps, teacher_acc, 'g-', linewidth=2.5, label='Teacher (FP16)',
            marker='D', markevery=15, markersize=5)
    ax2.plot(steps, student_acc, 'orange', linewidth=2.5, label='Student (INT4)',
            marker='o', markevery=15, markersize=5)
    ax2.fill_between(steps, student_acc, teacher_acc, alpha=0.2, color='red',
                     label='Accuracy Gap')

    ax2.set_xlabel('Training Step', fontsize=10, weight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=10, weight='bold')
    ax2.set_title('Teacher-Student Accuracy Convergence', fontsize=11, weight='bold')
    ax2.legend(loc='lower right', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2500)
    ax2.set_ylim(60, 82)

    # Annotate gap reduction
    ax2.annotate(f'Initial gap: {accuracy_gap[0]:.1f}%',
                xy=(0, student_acc[0]), xytext=(400, 68),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=8, bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
    ax2.annotate(f'Final gap: {accuracy_gap[-1]:.1f}%',
                xy=(2500, student_acc[-1]), xytext=(1800, 72),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=8, bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8))

    # Panel 3: Per-Layer Quantization Error
    ax3 = axes[1, 0]

    num_layers = 32
    layers = np.arange(num_layers)

    # Quantization error at different training stages
    error_step_0 = 0.014 + 0.003 * np.random.randn(num_layers) * 0.3
    error_step_500 = 0.011 + 0.002 * np.random.randn(num_layers) * 0.3
    error_step_2500 = 0.008 + 0.001 * np.random.randn(num_layers) * 0.3

    width = 0.25
    ax3.bar(layers - width, error_step_0, width, label='Step 0', color='#ff6b6b', alpha=0.8)
    ax3.bar(layers, error_step_500, width, label='Step 500', color='#feca57', alpha=0.8)
    ax3.bar(layers + width, error_step_2500, width, label='Step 2500', color='#48dbfb', alpha=0.8)

    ax3.set_xlabel('Layer Index', fontsize=10, weight='bold')
    ax3.set_ylabel('Relative Quantization Error', fontsize=10, weight='bold')
    ax3.set_title('Per-Layer Quantization Error Reduction', fontsize=11, weight='bold')
    ax3.legend(loc='upper right', frameon=True, shadow=True)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xlim(-1, num_layers)

    # Panel 4: Lambda Balance Ratio
    ax4 = axes[1, 1]

    # Ratio of quant_loss to kd_loss over training
    loss_ratio = (quant_loss / kd_loss) * 1000  # Scale for visibility
    ideal_ratio = np.full_like(steps, 1.0)  # Target ratio 1:1000

    ax4.plot(steps, loss_ratio, 'purple', linewidth=2.5, label='Actual Ratio (×1000)',
            marker='o', markevery=10, markersize=4)
    ax4.axhline(y=1.0, color='green', linestyle='--', linewidth=2,
               label='Target Ratio (1:1000)')
    ax4.fill_between(steps, 0.5, 1.5, alpha=0.2, color='green',
                    label='Safe Zone (0.5-1.5)')

    ax4.set_xlabel('Training Step', fontsize=10, weight='bold')
    ax4.set_ylabel('λ·𝓛_quant / 𝓛_KD (×1000)', fontsize=10, weight='bold')
    ax4.set_title('Lambda Balance Monitoring', fontsize=11, weight='bold')
    ax4.legend(loc='upper right', frameon=True, shadow=True)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 2500)
    ax4.set_ylim(0, 2)

    # Annotate warning zones
    ax4.axhspan(1.5, 2, alpha=0.15, color='red', label='_nolegend_')
    ax4.text(1250, 1.7, 'Too High: Forgetting risk', fontsize=8, ha='center',
            color='red', weight='bold')
    ax4.axhspan(0, 0.5, alpha=0.15, color='orange', label='_nolegend_')
    ax4.text(1250, 0.25, 'Too Low: No quant awareness', fontsize=8, ha='center',
            color='orange', weight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'qakd_training_dynamics.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: qakd_training_dynamics.png")

def create_quantization_comparison():
    """Compare different quantization approaches"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Quantization Methods Comparison', fontsize=14, weight='bold', y=1.00)

    # Left panel: Accuracy vs Memory Trade-off
    methods = ['Full\nPrecision', 'Standard\nDistillation', 'PTQ\nINT4', 'QAKD\nINT4']
    memory_gb = [14, 3, 1.75, 1.75]
    accuracy = [78.5, 72.4, 70.2, 76.3]
    colors = ['#28a745', '#17a2b8', '#ffc107', '#dc3545']

    ax1.scatter(memory_gb, accuracy, s=[500, 500, 500, 500], c=colors, alpha=0.7,
               edgecolors='black', linewidth=2)

    for i, method in enumerate(methods):
        ax1.annotate(method, (memory_gb[i], accuracy[i]),
                    textcoords="offset points", xytext=(0, 15), ha='center',
                    fontsize=10, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3))

    # Draw Pareto frontier
    pareto_x = [14, 1.75, 1.75]
    pareto_y = [78.5, 76.3, 76.3]
    ax1.plot(pareto_x, pareto_y, 'k--', linewidth=2, alpha=0.5, label='Pareto Frontier')

    ax1.set_xlabel('Memory (GB)', fontsize=11, weight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=11, weight='bold')
    ax1.set_title('Accuracy vs Memory Trade-off', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 16)
    ax1.set_ylim(68, 80)

    # Highlight QAKD advantage
    ax1.annotate('QAKD: Best accuracy\nat lowest memory!',
                xy=(1.75, 76.3), xytext=(6, 74),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
                fontsize=10, weight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # Right panel: Detailed Comparison Table
    ax2.axis('off')
    ax2.set_title('Detailed Method Comparison', fontsize=12, weight='bold', pad=20)

    table_data = [
        ['Method', 'Memory\n(GB)', 'Accuracy\n(%)', 'Speed\n(×)', 'Training\nTime', 'Accuracy\nRetention'],
        ['Full Precision', '14.0', '78.5', '1.0×', '-', '100%'],
        ['Standard KD', '3.0', '72.4', '3.5×', '~24h', '92%'],
        ['PTQ INT4', '1.75', '70.2', '7.0×', '~2h', '89%'],
        ['QAKD INT4', '1.75', '76.3', '7.0×', '~36h', '97%']
    ]

    table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0.05, 0.1, 0.9, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Style header row
    for i in range(6):
        cell = table[(0, i)]
        cell.set_facecolor('#4a90e2')
        cell.set_text_props(weight='bold', color='white')

    # Style data rows with alternating colors
    row_colors = ['#f0f0f0', '#ffffff', '#fff3cd', '#d4edda']
    for i in range(1, 5):
        for j in range(6):
            cell = table[(i, j)]
            cell.set_facecolor(row_colors[i-1])
            if i == 4:  # Highlight QAKD row
                cell.set_edgecolor('#28a745')
                cell.set_linewidth(2)

    # Add legend/notes
    ax2.text(0.5, 0.05,
            '* Speed measured on CPU inference | Training time on 1×A100 | 7B→1.5B compression',
            ha='center', fontsize=8, style='italic', transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig(output_dir / 'quantization_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: quantization_comparison.png")

def create_awq_impact():
    """Visualize AWQ activation-aware quantization impact"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('AWQ: Activation-Aware Weight Quantization Impact',
                 fontsize=14, weight='bold', y=1.00)

    # Left panel: Activation Magnitude Distribution
    np.random.seed(42)

    # Simulate activation magnitudes for 512 channels
    activations = np.abs(np.random.randn(512)) * 2
    activations[0:50] = activations[0:50] * 5  # Some channels have high activations

    sorted_acts = np.sort(activations)[::-1]
    channels = np.arange(512)

    # Color by importance
    colors = ['red' if act > 5 else 'orange' if act > 3 else 'green' for act in sorted_acts]

    ax1.bar(channels, sorted_acts, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Mark regions
    ax1.axhline(y=5, color='red', linestyle='--', linewidth=2, label='High Importance (>5)')
    ax1.axhline(y=3, color='orange', linestyle='--', linewidth=2, label='Medium Importance (3-5)')

    ax1.set_xlabel('Channel Index (sorted)', fontsize=10, weight='bold')
    ax1.set_ylabel('Activation Magnitude', fontsize=10, weight='bold')
    ax1.set_title('Per-Channel Activation Magnitude Distribution', fontsize=11, weight='bold')
    ax1.legend(loc='upper right', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xlim(0, 512)

    # Annotate
    ax1.annotate('Critical channels\n(fine quantization)',
                xy=(25, 10), xytext=(150, 11),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=9, weight='bold',
                bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
    ax1.annotate('Less critical\n(coarse quantization OK)',
                xy=(400, 1), xytext=(300, 4),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=9, weight='bold',
                bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8))

    # Right panel: Quantization Error Comparison
    categories = ['Uniform\nQuantization', 'AWQ\nQuantization']

    # Error breakdown by channel type
    high_importance = [0.025, 0.005]  # 80% reduction
    medium_importance = [0.012, 0.006]  # 50% reduction
    low_importance = [0.008, 0.007]  # 12.5% reduction

    x = np.arange(len(categories))
    width = 0.25

    bars1 = ax2.bar(x - width, high_importance, width, label='High Importance (top 10%)',
                   color='#ff6b6b', alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x, medium_importance, width, label='Medium Importance (10-20%)',
                   color='#feca57', alpha=0.8, edgecolor='black')
    bars3 = ax2.bar(x + width, low_importance, width, label='Low Importance (80%)',
                   color='#48dbfb', alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8, weight='bold')

    ax2.set_ylabel('Quantization Error', fontsize=10, weight='bold')
    ax2.set_title('Quantization Error by Channel Importance', fontsize=11, weight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=10, weight='bold')
    ax2.legend(loc='upper right', frameon=True, shadow=True, fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 0.03)

    # Add overall improvement text
    overall_uniform = np.mean([0.025, 0.012, 0.008])
    overall_awq = np.mean([0.005, 0.006, 0.007])
    improvement = (1 - overall_awq / overall_uniform) * 100

    ax2.text(0.5, 0.92, f'Overall Error Reduction: {improvement:.1f}%',
            ha='center', transform=ax2.transAxes, fontsize=11, weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / 'awq_impact.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: awq_impact.png")

def create_gptq_mechanism():
    """Visualize GPTQ Hessian-based quantization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('GPTQ: Hessian-Based Quantization with Error Compensation',
                 fontsize=14, weight='bold', y=1.00)

    # Left panel: Weight Distribution
    np.random.seed(42)

    # Simulate weight distribution (heavy-tailed with outliers)
    weights_normal = np.random.randn(9500) * 0.03
    weights_outliers = np.random.randn(500) * 0.15
    weights = np.concatenate([weights_normal, weights_outliers])

    ax1.hist(weights, bins=100, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)

    # Mark quantization boundaries for uniform INT4 (16 levels)
    quantization_levels = np.linspace(weights.min(), weights.max(), 16)
    for level in quantization_levels:
        ax1.axvline(x=level, color='red', linestyle='--', alpha=0.3, linewidth=1)

    ax1.set_xlabel('Weight Value', fontsize=10, weight='bold')
    ax1.set_ylabel('Frequency', fontsize=10, weight='bold')
    ax1.set_title('Heavy-Tailed Weight Distribution', fontsize=11, weight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Highlight outlier region
    ax1.axvspan(-0.5, -0.1, alpha=0.2, color='orange', label='Outliers (critical)')
    ax1.axvspan(0.1, 0.5, alpha=0.2, color='orange')
    ax1.legend(loc='upper right', frameon=True, shadow=True)

    # Annotate
    ax1.annotate('95% of weights\n(well-quantized)',
                xy=(0, 800), xytext=(0.08, 900),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=9, weight='bold',
                bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8))
    ax1.annotate('5% outliers\n(poorly quantized)',
                xy=(0.3, 50), xytext=(0.2, 400),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=9, weight='bold',
                bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))

    # Right panel: Sequential Quantization Process
    ax2.axis('off')
    ax2.set_title('GPTQ Sequential Quantization Algorithm', fontsize=12, weight='bold', pad=10)

    # Draw flowchart
    y_pos = 9
    box_height = 0.8
    box_width = 8

    # Step 1
    box1 = FancyBboxPatch((1, y_pos-box_height), box_width, box_height,
                          boxstyle="round,pad=0.1", facecolor='#e3f2fd',
                          edgecolor='#2196f3', linewidth=2)
    ax2.add_patch(box1)
    ax2.text(5, y_pos-box_height/2, 'Step 1: Compute Hessian H = X^T X\n(using calibration data)',
            ha='center', va='center', fontsize=9, weight='bold')

    y_pos -= 1.5
    ax2.annotate('', xy=(5, y_pos+0.2), xytext=(5, y_pos+0.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Step 2
    y_pos -= 0.5
    box2 = FancyBboxPatch((1, y_pos-box_height), box_width, box_height,
                          boxstyle="round,pad=0.1", facecolor='#fff3e0',
                          edgecolor='#ff9800', linewidth=2)
    ax2.add_patch(box2)
    ax2.text(5, y_pos-box_height/2, 'Step 2: Cholesky Decomposition H = LL^T\n(for stable inversion)',
            ha='center', va='center', fontsize=9, weight='bold')

    y_pos -= 1.5
    ax2.annotate('', xy=(5, y_pos+0.2), xytext=(5, y_pos+0.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Step 3
    y_pos -= 0.5
    box3 = FancyBboxPatch((1, y_pos-box_height), box_width, box_height,
                          boxstyle="round,pad=0.1", facecolor='#f3e5f5',
                          edgecolor='#9c27b0', linewidth=2)
    ax2.add_patch(box3)
    ax2.text(5, y_pos-box_height/2, 'Step 3: For each weight w_i (sequential):\nQuantize: w_q,i = Quantize(w_i)',
            ha='center', va='center', fontsize=9, weight='bold')

    y_pos -= 1.5
    ax2.annotate('', xy=(5, y_pos+0.2), xytext=(5, y_pos+0.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Step 4
    y_pos -= 0.5
    box4 = FancyBboxPatch((1, y_pos-box_height), box_width, box_height,
                          boxstyle="round,pad=0.1", facecolor='#ffebee',
                          edgecolor='#f44336', linewidth=2)
    ax2.add_patch(box4)
    ax2.text(5, y_pos-box_height/2, 'Step 4: Compute error: ε_i = w_i - w_q,i\nCompensate future weights',
            ha='center', va='center', fontsize=9, weight='bold')

    y_pos -= 1.5
    ax2.annotate('', xy=(5, y_pos+0.2), xytext=(5, y_pos+0.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Step 5
    y_pos -= 0.5
    box5 = FancyBboxPatch((1, y_pos-box_height), box_width, box_height,
                          boxstyle="round,pad=0.1", facecolor='#e8f5e9',
                          edgecolor='#4caf50', linewidth=2)
    ax2.add_patch(box5)
    ax2.text(5, y_pos-box_height/2, 'Step 5: Update remaining weights:\nw_j ← w_j - (H_ij / H_ii) × ε_i',
            ha='center', va='center', fontsize=9, weight='bold')

    # Add formula box
    formula_box = FancyBboxPatch((0.5, 0.2), 9, 1.2, boxstyle="round,pad=0.1",
                                 facecolor='#fff9c4', edgecolor='#f57f17', linewidth=2)
    ax2.add_patch(formula_box)
    ax2.text(5, 1.1, 'Key Formula (Error Compensation):', ha='center', fontsize=10, weight='bold')
    ax2.text(5, 0.7, 'Error(W_q) = Tr((W - W_q)^T H (W - W_q))',
            ha='center', fontsize=9, family='monospace')
    ax2.text(5, 0.4, 'Minimize error by compensating in Hessian-aware manner',
            ha='center', fontsize=8, style='italic')

    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    plt.tight_layout()
    plt.savefig(output_dir / 'gptq_mechanism.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: gptq_mechanism.png")

def create_mixed_precision_strategy():
    """Visualize mixed-precision training strategy"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Mixed-Precision Training Strategy for QAKD',
                 fontsize=14, weight='bold', y=1.00)

    # Left panel: Precision Hierarchy
    ax1.axis('off')
    ax1.set_title('Precision Hierarchy in QAKD Training', fontsize=12, weight='bold', pad=10)

    operations = [
        ('Forward (Inference)', 'INT4/INT8', '#e8f5e9', '#4caf50'),
        ('Forward (Training)', 'FP16', '#e3f2fd', '#2196f3'),
        ('Gradients', 'FP16', '#fff3e0', '#ff9800'),
        ('Weight Updates', 'FP32', '#f3e5f5', '#9c27b0'),
        ('Normalization', 'FP32', '#ffebee', '#f44336'),
        ('Loss Computation', 'FP32', '#fce4ec', '#e91e63')
    ]

    y_pos = 9
    for op, precision, bgcolor, edgecolor in operations:
        box = FancyBboxPatch((1, y_pos-0.8), 8, 0.7, boxstyle="round,pad=0.1",
                            facecolor=bgcolor, edgecolor=edgecolor, linewidth=2.5)
        ax1.add_patch(box)
        ax1.text(3, y_pos-0.45, op, ha='left', va='center', fontsize=10, weight='bold')
        ax1.text(7.5, y_pos-0.45, precision, ha='center', va='center',
                fontsize=10, weight='bold', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.5))
        y_pos -= 1.2

    # Add legend for benefits
    legend_y = 1.5
    ax1.text(5, legend_y, 'Benefits:', ha='center', fontsize=11, weight='bold')
    ax1.text(5, legend_y-0.5, '• FP16: 2-3× faster, half memory', ha='center', fontsize=9)
    ax1.text(5, legend_y-0.9, '• FP32: Prevents underflow, stable updates', ha='center', fontsize=9)
    ax1.text(5, legend_y-1.3, '• INT4/8: 4-8× memory reduction for inference', ha='center', fontsize=9)

    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)

    # Right panel: Gradient Scaling Mechanism
    ax2.set_title('Gradient Scaling to Prevent Underflow', fontsize=12, weight='bold')

    # Simulate gradient magnitudes
    np.random.seed(42)
    step_numbers = np.arange(100)

    # Without scaling (many underflows)
    gradients_unscaled = np.abs(np.random.randn(100)) * 1e-7
    underflow_threshold = 6e-8  # FP16 minimum

    # With scaling (S=1024)
    gradients_scaled = gradients_unscaled * 1024

    # Plot
    ax2.semilogy(step_numbers, gradients_unscaled, 'r-', alpha=0.5, linewidth=1,
                label='Without Scaling')
    ax2.semilogy(step_numbers, gradients_scaled, 'g-', alpha=0.7, linewidth=2,
                label='With Scaling (S=1024)')
    ax2.axhline(y=underflow_threshold, color='red', linestyle='--', linewidth=2.5,
               label='FP16 Underflow Threshold (6×10⁻⁸)')

    # Shade underflow region
    ax2.fill_between(step_numbers, 1e-10, underflow_threshold, alpha=0.2, color='red',
                    label='Underflow Zone')
    ax2.fill_between(step_numbers, underflow_threshold, 1e-2, alpha=0.2, color='green',
                    label='Safe Zone')

    ax2.set_xlabel('Training Step', fontsize=10, weight='bold')
    ax2.set_ylabel('Gradient Magnitude', fontsize=10, weight='bold')
    ax2.legend(loc='upper right', frameon=True, shadow=True, fontsize=8)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_ylim(1e-10, 1e-2)

    # Annotate
    ax2.annotate('Without scaling:\nMost gradients → 0',
                xy=(50, 3e-8), xytext=(25, 1e-6),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=9, weight='bold',
                bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
    ax2.annotate('With scaling:\nGradients preserved',
                xy=(50, 3e-5), xytext=(70, 1e-4),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=9, weight='bold',
                bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / 'mixed_precision_strategy.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: mixed_precision_strategy.png")

def create_deployment_workflow():
    """Visualize QAKD training to deployment workflow"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')
    fig.suptitle('QAKD: Training to Deployment Workflow', fontsize=16, weight='bold', y=0.98)

    # Phase 1: Model Selection
    phase1_box = FancyBboxPatch((0.5, 10), 4, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#e8f5e9', edgecolor='#4caf50', linewidth=3)
    ax.add_patch(phase1_box)
    ax.text(2.5, 11.2, 'Phase 1: Model Selection', fontsize=11, weight='bold', ha='center')
    ax.text(2.5, 10.8, 'Teacher: 7B FP16', fontsize=9, ha='center')
    ax.text(2.5, 10.4, 'Student: 1.5B FP32', fontsize=9, ha='center')

    # Arrow to Phase 2
    ax.annotate('', xy=(5, 10.75), xytext=(4.6, 10.75),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    # Phase 2: Hyperparameter Config
    phase2_box = FancyBboxPatch((5, 10), 4, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#e3f2fd', edgecolor='#2196f3', linewidth=3)
    ax.add_patch(phase2_box)
    ax.text(7, 11.2, 'Phase 2: Configuration', fontsize=11, weight='bold', ha='center')
    ax.text(7, 10.8, 'τ=3, λ=5e-4, bits=4', fontsize=9, ha='center', family='monospace')
    ax.text(7, 10.4, 'LR=2e-5, BS=16', fontsize=9, ha='center', family='monospace')

    # Arrow to Phase 3
    ax.annotate('', xy=(9.5, 10.75), xytext=(9.1, 10.75),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    # Phase 3: AWQ Calibration
    phase3_box = FancyBboxPatch((9.5, 10), 4, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#fff3e0', edgecolor='#ff9800', linewidth=3)
    ax.add_patch(phase3_box)
    ax.text(11.5, 11.2, 'Phase 3: AWQ Calibration', fontsize=11, weight='bold', ha='center')
    ax.text(11.5, 10.8, 'Collect activations', fontsize=9, ha='center')
    ax.text(11.5, 10.4, '500-1000 samples', fontsize=9, ha='center')

    # Arrow down to Phase 4
    ax.annotate('', xy=(7, 9.4), xytext=(7, 9.9),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    # Phase 4: QAKD Training (large center box)
    phase4_box = FancyBboxPatch((1, 5.5), 12, 3.5, boxstyle="round,pad=0.1",
                                facecolor='#f3e5f5', edgecolor='#9c27b0', linewidth=3)
    ax.add_patch(phase4_box)
    ax.text(7, 8.7, 'Phase 4: QAKD Training (3 epochs, ~36 hours)',
            fontsize=12, weight='bold', ha='center')

    # Training loop components
    components = [
        ('Teacher Forward\n(FP16)', 2, 7.5, '#d4edda'),
        ('Student Forward\n(Fake Quant)', 5, 7.5, '#fff3cd'),
        ('𝓛_KD Loss', 8, 7.5, '#ffcccc'),
        ('𝓛_quant Loss', 11, 7.5, '#cce5ff'),
        ('Backward\n(STE)', 3.5, 6.3, '#ffe6cc'),
        ('Update Weights\n(FP32)', 7, 6.3, '#ccffcc'),
        ('Checkpoint', 10.5, 6.3, '#e6ccff')
    ]

    for label, x, y, color in components:
        comp_box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.7, boxstyle="round,pad=0.05",
                                  facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(comp_box)
        ax.text(x, y, label, fontsize=8, ha='center', va='center', weight='bold')

    # Arrow down to Phase 5
    ax.annotate('', xy=(7, 5), xytext=(7, 5.4),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    # Phase 5: Post-Training Quantization
    phase5_box = FancyBboxPatch((2, 3.5), 10, 1.3, boxstyle="round,pad=0.1",
                                facecolor='#ffebee', edgecolor='#f44336', linewidth=3)
    ax.add_patch(phase5_box)
    ax.text(7, 4.5, 'Phase 5: True Quantization (GPTQ)', fontsize=11, weight='bold', ha='center')
    ax.text(7, 4.1, 'Convert FP32 → INT4 using Hessian-aware quantization',
            fontsize=9, ha='center', style='italic')
    ax.text(7, 3.8, 'Memory: 3GB → 1.75GB', fontsize=9, ha='center', color='green', weight='bold')

    # Arrow down to Phase 6
    ax.annotate('', xy=(7, 3), xytext=(7, 3.4),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    # Phase 6: Evaluation
    phase6_box = FancyBboxPatch((2, 1.5), 4.5, 1.3, boxstyle="round,pad=0.1",
                                facecolor='#fff9c4', edgecolor='#f57f17', linewidth=3)
    ax.add_patch(phase6_box)
    ax.text(4.25, 2.5, 'Phase 6: Evaluation', fontsize=11, weight='bold', ha='center')
    ax.text(4.25, 2.1, 'MMLU, HellaSwag, etc.', fontsize=9, ha='center')
    ax.text(4.25, 1.8, 'Target: ≥97% teacher acc', fontsize=9, ha='center', weight='bold')

    # Arrow to Phase 7
    ax.annotate('', xy=(7, 2.15), xytext=(6.6, 2.15),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    # Phase 7: Deployment
    phase7_box = FancyBboxPatch((7.5, 1.5), 4.5, 1.3, boxstyle="round,pad=0.1",
                                facecolor='#c8e6c9', edgecolor='#2e7d32', linewidth=3)
    ax.add_patch(phase7_box)
    ax.text(9.75, 2.5, 'Phase 7: Deployment', fontsize=11, weight='bold', ha='center')
    ax.text(9.75, 2.1, 'Edge devices, Mobile', fontsize=9, ha='center')
    ax.text(9.75, 1.8, '7× speedup, 1.75GB', fontsize=9, ha='center', weight='bold', color='green')

    # Timeline annotations
    ax.text(0.3, 9.5, '~2h', fontsize=9, style='italic', color='gray')
    ax.text(0.3, 7, '36h', fontsize=9, style='italic', color='gray', weight='bold')
    ax.text(0.3, 4.2, '~2h', fontsize=9, style='italic', color='gray')
    ax.text(0.3, 2.2, '~4h', fontsize=9, style='italic', color='gray')

    # Success metrics box
    metrics_box = FancyBboxPatch((1, 0.2), 12, 1, boxstyle="round,pad=0.1",
                                 facecolor='#e1f5dd', edgecolor='#388e3c', linewidth=2.5)
    ax.add_patch(metrics_box)
    ax.text(7, 0.9, 'Success Metrics', fontsize=11, weight='bold', ha='center')
    ax.text(7, 0.6, 'Accuracy: 76.3% (97% of teacher 78.5%) | Memory: 1.75GB (8× reduction) | Speed: 7× faster on CPU',
            fontsize=9, ha='center', weight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'deployment_workflow.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: deployment_workflow.png")

# Main execution
if __name__ == "__main__":
    print("Generating QAKD visualizations...")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    create_qakd_framework()
    create_training_dynamics()
    create_quantization_comparison()
    create_awq_impact()
    create_gptq_mechanism()
    create_mixed_precision_strategy()
    create_deployment_workflow()

    print("-" * 60)
    print(f"✓ All QAKD visualizations generated successfully!")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Total files: 7 PNG images (300 DPI)")
