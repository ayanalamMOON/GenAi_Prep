"""
Visualization 2 of 8 for Section 14: Training Infrastructure & FLOPs Analysis

This script visualizes:
- FLOPs calculation breakdown (forward/backward pass)
- Training time vs model size comparison

Output: visualization/images/llmops/training_infrastructure.png
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
print("SECTION 14 - VISUALIZATION 2/8: Training Infrastructure & FLOPs")
print("=" * 80)

def create_training_infrastructure():
    """Two-panel visualization: FLOPs breakdown and training time analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Training Infrastructure: Compute Requirements & Time Estimation',
                 fontsize=16, fontweight='bold', y=1.02)

    # Panel 1: FLOPs Breakdown
    ax1 = axes[0]

    # Model sizes (parameters)
    models = ['125M', '350M', '760M', '1.5B', '3B', '7B', '13B', '30B', '70B']
    params = np.array([0.125, 0.35, 0.76, 1.5, 3, 7, 13, 30, 70])  # billions
    tokens = 20  # billion tokens (Chinchilla optimal)

    # FLOPs calculation: 6PT (P=params, T=tokens)
    total_flops = 6 * params * tokens  # in E18 (exaflops)
    forward_flops = 2 * params * tokens
    backward_flops = 4 * params * tokens

    # Stacked bar chart
    x = np.arange(len(models))
    width = 0.6

    ax1.bar(x, forward_flops, width, label='Forward Pass (2PT FLOPs)',
           color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.2)
    ax1.bar(x, backward_flops, width, bottom=forward_flops,
           label='Backward Pass (4PT FLOPs)',
           color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.2)

    ax1.set_xlabel('Model Size (Parameters)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Total FLOPs (×10¹⁸)', fontweight='bold', fontsize=12)
    ax1.set_title('FLOPs Breakdown: Forward vs Backward Pass\n(Chinchilla Scaling: 20B tokens)',
                  fontweight='bold', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=11, loc='upper left')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add total FLOPs labels on top
    for i, (f, b) in enumerate(zip(forward_flops, backward_flops)):
        total = f + b
        if total < 100:
            ax1.text(i, total + 10, f'{total:.0f}E18', ha='center',
                    fontsize=9, fontweight='bold')
        else:
            ax1.text(i, total + 30, f'{total:.0f}E18', ha='center',
                    fontsize=9, fontweight='bold')

    # Add Chinchilla formula annotation
    ax1.text(0.5, 0.95, 'Chinchilla Formula: Total FLOPs = 6PT\n' +
             'P = parameters (billions), T = tokens (billions)',
             transform=ax1.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
             fontsize=9, style='italic')

    # Panel 2: Training Time Analysis
    ax2 = axes[1]

    # Training time calculation (assuming A100 GPUs: 312 TFLOPS, MFU=40%)
    gpu_tflops = 312  # A100
    mfu = 0.40  # Model FLOPs Utilization
    effective_tflops_per_gpu = gpu_tflops * mfu

    # Different GPU configurations
    gpu_configs = [8, 16, 32, 64, 128]
    colors_gpu = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#3498db']

    for gpu_count, color in zip(gpu_configs, colors_gpu):
        # Training time in days for each model
        total_tflops = total_flops * 1e6  # Convert E18 to TFLOP
        training_seconds = total_tflops / (effective_tflops_per_gpu * gpu_count)
        training_days = training_seconds / (3600 * 24)

        ax2.plot(params, training_days, marker='o', markersize=8, linewidth=2.5,
                label=f'{gpu_count} A100 GPUs', color=color, alpha=0.85)

    ax2.set_xlabel('Model Size (Billion Parameters)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Training Time (Days)', fontweight='bold', fontsize=12)
    ax2.set_title('Training Time Estimation: GPU Count Impact\n(40% MFU, Chinchilla-optimal tokens)',
                  fontweight='bold', fontsize=13)
    ax2.set_yscale('log')
    ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax2.grid(True, alpha=0.3, which='both', linestyle='--')

    # Add reference lines
    ax2.axhline(y=7, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax2.text(35, 8, 'Week (7 days)', fontsize=9, color='red', fontweight='bold')

    ax2.axhline(y=30, color='orange', linestyle='--', linewidth=2, alpha=0.5)
    ax2.text(35, 35, 'Month (30 days)', fontsize=9, color='orange', fontweight='bold')

    # Add efficiency note
    ax2.text(0.05, 0.05, 'MFU (Model FLOPs Utilization):\n' +
             '• Communication overhead\n• Memory bandwidth limits\n' +
             '• Kernel launch delays\n\nTypical: 30-50% for large models',
             transform=ax2.transAxes, ha='left', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3),
             fontsize=8)

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/training_infrastructure.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    create_training_infrastructure()
    print(f"\n✓ Section 14 Visualization 2/8 completed!")
    print(f"✓ Output: {OUTPUT_DIR}/training_infrastructure.png")
