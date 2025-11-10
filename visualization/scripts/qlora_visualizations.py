"""
QLoRA Visualization Script
Generates publication-quality visualizations for QLoRA section of LLM Study Material
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
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

# Output directory
output_dir = Path(__file__).parent.parent / 'images' / 'qlora'
output_dir.mkdir(parents=True, exist_ok=True)

def create_qlora_architecture():
    """QLoRA architecture showing 4-bit quantization + LoRA adapters"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7, 7.5, 'QLoRA Architecture: 4-bit Quantized Base + 16-bit LoRA Adapters',
            fontsize=14, weight='bold', ha='center')

    # Base model (quantized)
    base_box = FancyBboxPatch((0.5, 3), 4, 3, boxstyle="round,pad=0.1",
                              facecolor='#e8f4f8', edgecolor='#2E86AB', linewidth=2)
    ax.add_patch(base_box)
    ax.text(2.5, 5.5, 'Frozen Base Model', fontsize=11, weight='bold', ha='center')
    ax.text(2.5, 5, 'Weights: 4-bit NF4', fontsize=9, ha='center', style='italic')
    ax.text(2.5, 4.5, 'W ∈ {0,1,...,15}', fontsize=9, ha='center', family='monospace')
    ax.text(2.5, 4, 'Memory: 0.5 bytes/param', fontsize=8, ha='center', color='green')
    ax.text(2.5, 3.5, '7B model → 3.5 GB', fontsize=8, ha='center', weight='bold', color='green')

    # LoRA adapters
    lora_a = FancyBboxPatch((5.5, 4.5), 2.5, 1.3, boxstyle="round,pad=0.05",
                            facecolor='#fff3cd', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(lora_a)
    ax.text(6.75, 5.5, 'LoRA Matrix A', fontsize=10, weight='bold', ha='center')
    ax.text(6.75, 5.1, 'd × r (16-bit)', fontsize=8, ha='center', style='italic')

    lora_b = FancyBboxPatch((5.5, 3), 2.5, 1.3, boxstyle="round,pad=0.05",
                            facecolor='#fff3cd', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(lora_b)
    ax.text(6.75, 3.8, 'LoRA Matrix B', fontsize=10, weight='bold', ha='center')
    ax.text(6.75, 3.4, 'r × d (16-bit)', fontsize=8, ha='center', style='italic')

    # Output combination
    output_box = FancyBboxPatch((9, 3.5), 4, 2, boxstyle="round,pad=0.1",
                                facecolor='#d4edda', edgecolor='#28a745', linewidth=2)
    ax.add_patch(output_box)
    ax.text(11, 5, 'Combined Output', fontsize=11, weight='bold', ha='center')
    ax.text(11, 4.5, 'W_quantized · x + B·A·x', fontsize=9, ha='center', family='monospace')
    ax.text(11, 4, 'Full precision during forward', fontsize=8, ha='center', style='italic')

    # Arrows
    ax.annotate('', xy=(5.3, 5.2), xytext=(4.5, 5.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='#2E86AB'))
    ax.annotate('', xy=(5.3, 3.6), xytext=(4.5, 3.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='#2E86AB'))
    ax.annotate('', xy=(8.8, 5.1), xytext=(8.1, 5.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='#f39c12'))
    ax.annotate('', xy=(8.8, 3.7), xytext=(8.1, 3.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='#f39c12'))

    # Memory breakdown (bottom)
    ax.text(2.5, 2.3, 'Memory Breakdown (7B Model)', fontsize=11, weight='bold', ha='center')
    ax.text(2.5, 1.9, '⚬ Base (4-bit): 3.5 GB', fontsize=9, ha='center', color='#2E86AB')
    ax.text(2.5, 1.5, '⚬ LoRA (r=8): ~32 MB', fontsize=9, ha='center', color='#f39c12')
    ax.text(2.5, 1.1, '⚬ Gradients: ~64 MB', fontsize=9, ha='center', color='gray')
    ax.text(2.5, 0.7, '⚬ Total: ~3.6 GB', fontsize=10, ha='center', weight='bold', color='green')

    # Key innovations (right)
    ax.text(11, 2.5, 'QLoRA Innovations', fontsize=11, weight='bold', ha='center')
    ax.text(11, 2.1, '1. 4-bit NormalFloat (NF4)', fontsize=9, ha='center')
    ax.text(11, 1.7, '2. Double Quantization', fontsize=9, ha='center')
    ax.text(11, 1.3, '3. Paged Optimizers', fontsize=9, ha='center')
    ax.text(11, 0.9, '4. 16-bit Compute', fontsize=9, ha='center')

    plt.tight_layout()
    plt.savefig(output_dir / 'qlora_architecture.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: qlora_architecture.png")

def create_quantization_comparison():
    """Compare different quantization schemes"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Bit precision memory comparison
    methods = ['FP32', 'FP16', 'INT8', 'NF4']
    memory_7b = [28, 14, 7, 3.5]  # GB for 7B model
    colors = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']

    bars = ax1.barh(methods, memory_7b, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('GPU Memory (GB)', fontsize=11, weight='bold')
    ax1.set_title('Memory Usage by Precision (7B Model)', fontsize=12, weight='bold')
    ax1.grid(axis='x', alpha=0.3)

    for i, (bar, mem) in enumerate(zip(bars, memory_7b)):
        ax1.text(mem + 0.5, i, f'{mem} GB', va='center', fontsize=10, weight='bold')

    # 2. Quantization error comparison
    methods_q = ['FP16\n(baseline)', 'INT8\n(uniform)', 'INT4\n(uniform)', 'NF4\n(optimal)']
    errors = [0.0, 2.8, 8.5, 3.2]  # Relative error percentage

    bars = ax2.bar(methods_q, errors, color=['#95a5a6', '#3498db', '#e67e22', '#27ae60'],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Quantization Error (%)', fontsize=11, weight='bold')
    ax2.set_title('Quantization Error Comparison', fontsize=12, weight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Acceptable threshold')
    ax2.legend()

    for bar, err in zip(bars, errors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{err}%', ha='center', va='bottom', fontsize=9, weight='bold')

    # 3. NF4 bins visualization
    # Standard normal distribution
    x = np.linspace(-3, 3, 1000)
    normal_dist = (1/np.sqrt(2*np.pi)) * np.exp(-x**2/2)

    # NF4 bins (optimal for normal distribution)
    nf4_bins = [-1.0, -0.6961, -0.5250, -0.3949, -0.2844, -0.1848,
                -0.0911, 0.0, 0.0796, 0.1609, 0.2461, 0.3379,
                0.4407, 0.5626, 0.7229, 1.0]

    ax3.plot(x, normal_dist, 'b-', linewidth=2, label='N(0,1) distribution')

    # Draw bins
    for i in range(len(nf4_bins)-1):
        ax3.axvline(nf4_bins[i], color='red', linestyle='--', alpha=0.6, linewidth=1)
    ax3.axvline(nf4_bins[-1], color='red', linestyle='--', alpha=0.6, linewidth=1, label='NF4 bin edges')

    # Shade regions
    for i in range(len(nf4_bins)-1):
        mask = (x >= nf4_bins[i]) & (x < nf4_bins[i+1])
        ax3.fill_between(x[mask], 0, normal_dist[mask], alpha=0.2, color=f'C{i%10}')

    ax3.set_xlabel('Weight Value', fontsize=11, weight='bold')
    ax3.set_ylabel('Probability Density', fontsize=11, weight='bold')
    ax3.set_title('NF4: Information-Theoretic Optimal Bins', fontsize=12, weight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_xlim(-1.5, 1.5)

    # 4. Model size vs achievable max model size
    gpu_memory = [8, 16, 24, 40, 48, 80]
    max_fp16 = [0.5, 1, 1.5, 2.5, 3, 5]  # Billion params
    max_qlora = [2, 4, 7, 13, 15, 25]  # Billion params

    ax4.plot(gpu_memory, max_fp16, 'o-', linewidth=2.5, markersize=8,
             label='Standard LoRA (FP16)', color='#3498db')
    ax4.plot(gpu_memory, max_qlora, 's-', linewidth=2.5, markersize=8,
             label='QLoRA (4-bit NF4)', color='#27ae60')

    ax4.fill_between(gpu_memory, max_fp16, max_qlora, alpha=0.2, color='green')

    ax4.set_xlabel('GPU Memory (GB)', fontsize=11, weight='bold')
    ax4.set_ylabel('Max Model Size (Billion Params)', fontsize=11, weight='bold')
    ax4.set_title('Trainable Model Size vs GPU Memory', fontsize=12, weight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # Add annotations
    ax4.annotate('4× improvement', xy=(24, 3.5), xytext=(30, 5),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=10, weight='bold', color='green')

    plt.tight_layout()
    plt.savefig(output_dir / 'quantization_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: quantization_comparison.png")

def create_double_quantization():
    """Visualize double quantization mechanism"""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')

    ax.text(7, 6.5, 'Double Quantization: Quantizing the Quantization Constants',
            fontsize=14, weight='bold', ha='center')

    # Step 1: Original weights
    step1 = FancyBboxPatch((0.5, 4), 3, 1.5, boxstyle="round,pad=0.1",
                           facecolor='#e8f4f8', edgecolor='#2E86AB', linewidth=2)
    ax.add_patch(step1)
    ax.text(2, 5.2, 'Step 1: FP16 Weights', fontsize=11, weight='bold', ha='center')
    ax.text(2, 4.7, 'W ∈ ℝ (16-bit)', fontsize=9, ha='center')
    ax.text(2, 4.3, 'Memory: 2 bytes/param', fontsize=8, ha='center', color='red')

    # Arrow
    ax.annotate('Quantize', xy=(3.8, 4.75), xytext=(4.2, 4.75),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='black'),
                fontsize=10, weight='bold', ha='center')

    # Step 2: First quantization
    step2 = FancyBboxPatch((4.5, 4), 3.5, 1.5, boxstyle="round,pad=0.1",
                           facecolor='#fff3cd', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(step2)
    ax.text(6.25, 5.2, 'Step 2: 4-bit Quantization', fontsize=11, weight='bold', ha='center')
    ax.text(6.25, 4.7, 'W_q = scale₁ × W_nf4', fontsize=9, ha='center', family='monospace')
    ax.text(6.25, 4.3, 'Weights: 0.5 bytes/param', fontsize=8, ha='center', color='green')
    ax.text(6.25, 3.85, 'Scale₁: FP32 (4 bytes/block)', fontsize=8, ha='center', color='red', style='italic')

    # Arrow
    ax.annotate('Quantize\nscale', xy=(8.3, 4.75), xytext=(8.7, 4.75),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='black'),
                fontsize=10, weight='bold', ha='center')

    # Step 3: Double quantization
    step3 = FancyBboxPatch((9, 4), 4.5, 1.5, boxstyle="round,pad=0.1",
                           facecolor='#d4edda', edgecolor='#28a745', linewidth=2)
    ax.add_patch(step3)
    ax.text(11.25, 5.2, 'Step 3: Double Quantization', fontsize=11, weight='bold', ha='center')
    ax.text(11.25, 4.7, 'W_q = (scale₂ × scale₁_8bit) × W_nf4', fontsize=9, ha='center', family='monospace')
    ax.text(11.25, 4.3, 'Weights: 0.5 bytes/param', fontsize=8, ha='center', color='green')
    ax.text(11.25, 3.85, 'Scale₁: 8-bit (1 byte/block)', fontsize=8, ha='center', color='green', style='italic')

    # Memory calculation
    ax.text(7, 3.2, 'Memory Savings for 65B Model:', fontsize=12, weight='bold', ha='center')

    # Table
    y_start = 2.5
    methods = ['Weights (4-bit)', 'Scale FP32', 'Scale 8-bit', 'Total', 'Savings']
    mem_before = ['32.5 GB', '4.0 GB', '-', '36.5 GB', '-']
    mem_after = ['32.5 GB', '-', '1.0 GB', '33.5 GB', '3.0 GB (8.2%)']

    ax.text(3.5, y_start, 'Component', fontsize=10, weight='bold', ha='center')
    ax.text(6.5, y_start, 'Single Quant', fontsize=10, weight='bold', ha='center')
    ax.text(9.5, y_start, 'Double Quant', fontsize=10, weight='bold', ha='center')

    for i, (method, before, after) in enumerate(zip(methods, mem_before, mem_after)):
        y = y_start - 0.35 * (i + 1)

        # Highlight savings row
        if i == 4:
            rect = Rectangle((2, y - 0.15), 9, 0.3, facecolor='#d4edda',
                           edgecolor='#28a745', linewidth=1.5, alpha=0.5)
            ax.add_patch(rect)

        ax.text(3.5, y, method, fontsize=9, ha='center')
        ax.text(6.5, y, before, fontsize=9, ha='center', family='monospace')
        ax.text(9.5, y, after, fontsize=9, ha='center', family='monospace',
               weight='bold' if i == 4 else 'normal')

    # Key insight
    insight_box = FancyBboxPatch((0.5, 0.1), 13, 0.5, boxstyle="round,pad=0.05",
                                 facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=2)
    ax.add_patch(insight_box)
    ax.text(7, 0.35, 'Key Insight: Quantize the scaling factors from FP32 (4 bytes) to 8-bit (1 byte) → Additional 8% memory reduction',
            fontsize=10, ha='center', style='italic', weight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'double_quantization.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: double_quantization.png")

def create_memory_comparison():
    """Memory comparison across different methods"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Stacked bar chart for memory components
    methods = ['Full\nFine-Tune', 'LoRA\n(FP16)', 'QLoRA\n(4-bit)']
    model_sizes = ['7B', '13B', '30B', '65B']

    # Memory breakdown for 7B model
    base_weights = [14, 14, 3.5]
    gradients = [14, 2, 0.5]
    optimizer = [28, 4, 1]

    x = np.arange(len(methods))
    width = 0.6

    p1 = ax1.bar(x, base_weights, width, label='Base Weights', color='#3498db')
    p2 = ax1.bar(x, gradients, width, bottom=base_weights, label='Gradients', color='#e74c3c')
    p3 = ax1.bar(x, optimizer, width, bottom=np.array(base_weights)+np.array(gradients),
                 label='Optimizer States', color='#f39c12')

    ax1.set_ylabel('GPU Memory (GB)', fontsize=11, weight='bold')
    ax1.set_title('Memory Breakdown (7B Model)', fontsize=12, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', alpha=0.3)

    # Add total labels
    totals = [sum(x) for x in zip(base_weights, gradients, optimizer)]
    for i, (total, method) in enumerate(zip(totals, methods)):
        ax1.text(i, total + 1, f'{total} GB', ha='center', va='bottom',
                fontsize=10, weight='bold')

    # Add reduction annotations
    ax1.annotate('', xy=(1, 20), xytext=(0, 56),
                arrowprops=dict(arrowstyle='<->', lw=2, color='green'))
    ax1.text(0.5, 38, '64% less', rotation=90, va='center',
            fontsize=10, weight='bold', color='green')

    ax1.annotate('', xy=(2, 5), xytext=(1, 20),
                arrowprops=dict(arrowstyle='<->', lw=2, color='green'))
    ax1.text(1.5, 12.5, '75% less', rotation=90, va='center',
            fontsize=10, weight='bold', color='green')

    # Right: Maximum trainable model size
    gpu_configs = ['GTX 1080 Ti\n(11 GB)', 'RTX 3090\n(24 GB)',
                   'A100 40GB', 'A100 80GB']
    max_full = [0, 0.5, 1, 2]  # Billion params
    max_lora = [0.5, 1.5, 3, 7]
    max_qlora = [3, 7, 13, 25]

    x2 = np.arange(len(gpu_configs))
    width2 = 0.25

    ax2.bar(x2 - width2, max_full, width2, label='Full Fine-Tune', color='#e74c3c', alpha=0.8)
    ax2.bar(x2, max_lora, width2, label='LoRA (FP16)', color='#3498db', alpha=0.8)
    ax2.bar(x2 + width2, max_qlora, width2, label='QLoRA (4-bit)', color='#27ae60', alpha=0.8)

    ax2.set_ylabel('Max Model Size (Billion Params)', fontsize=11, weight='bold')
    ax2.set_title('Maximum Trainable Model Size by GPU', fontsize=12, weight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(gpu_configs)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, gpu in enumerate(gpu_configs):
        if max_full[i] > 0:
            ax2.text(i - width2, max_full[i] + 0.3, f'{max_full[i]:.1f}B',
                    ha='center', fontsize=8)
        ax2.text(i, max_lora[i] + 0.3, f'{max_lora[i]:.1f}B',
                ha='center', fontsize=8)
        ax2.text(i + width2, max_qlora[i] + 0.5, f'{max_qlora[i]:.0f}B',
                ha='center', fontsize=8, weight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'memory_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: memory_comparison.png")

def create_paged_optimizers():
    """Visualize paged optimizer memory management"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.5, 'Paged Optimizers: Automatic GPU ↔ CPU Memory Management',
            fontsize=14, weight='bold', ha='center')

    # GPU Memory section
    gpu_box = FancyBboxPatch((0.5, 4), 6, 2.5, boxstyle="round,pad=0.1",
                             facecolor='#e8f4f8', edgecolor='#2E86AB', linewidth=3)
    ax.add_patch(gpu_box)
    ax.text(3.5, 6.2, 'GPU Memory (Limited)', fontsize=12, weight='bold', ha='center')

    # Active states
    active1 = Rectangle((1, 5), 2, 0.6, facecolor='#27ae60', edgecolor='black', linewidth=1.5)
    ax.add_patch(active1)
    ax.text(2, 5.3, 'Active Gradients', fontsize=9, ha='center', color='white', weight='bold')

    active2 = Rectangle((3.5, 5), 2, 0.6, facecolor='#27ae60', edgecolor='black', linewidth=1.5)
    ax.add_patch(active2)
    ax.text(4.5, 5.3, 'Hot Optimizer States', fontsize=9, ha='center', color='white', weight='bold')

    # Free space
    free = Rectangle((1, 4.3), 4.5, 0.5, facecolor='#ecf0f1', edgecolor='gray',
                     linewidth=1, linestyle='--')
    ax.add_patch(free)
    ax.text(3.25, 4.55, 'Free Space (~20%)', fontsize=9, ha='center', style='italic')

    # CPU Memory section
    cpu_box = FancyBboxPatch((7.5, 4), 6, 2.5, boxstyle="round,pad=0.1",
                             facecolor='#fff3cd', edgecolor='#f39c12', linewidth=3)
    ax.add_patch(cpu_box)
    ax.text(10.5, 6.2, 'CPU Memory (Abundant)', fontsize=12, weight='bold', ha='center')

    # Paged out states
    paged1 = Rectangle((8, 5), 2, 0.6, facecolor='#95a5a6', edgecolor='black', linewidth=1.5)
    ax.add_patch(paged1)
    ax.text(9, 5.3, 'Paged Momentum', fontsize=9, ha='center', color='white', weight='bold')

    paged2 = Rectangle((10.5, 5), 2, 0.6, facecolor='#95a5a6', edgecolor='black', linewidth=1.5)
    ax.add_patch(paged2)
    ax.text(11.5, 5.3, 'Paged Variance', fontsize=9, ha='center', color='white', weight='bold')

    # Large available space
    available = Rectangle((8, 4.3), 4.5, 0.5, facecolor='#d4edda', edgecolor='gray', linewidth=1)
    ax.add_patch(available)
    ax.text(10.25, 4.55, 'Available Space (100+ GB)', fontsize=9, ha='center', style='italic')

    # Bidirectional arrows
    ax.annotate('', xy=(7.2, 5.7), xytext=(6.7, 5.7),
                arrowprops=dict(arrowstyle='->', lw=3, color='#e74c3c'))
    ax.text(6.95, 6, 'Page Out', fontsize=9, ha='center', color='#e74c3c', weight='bold')

    ax.annotate('', xy=(6.7, 5.1), xytext=(7.2, 5.1),
                arrowprops=dict(arrowstyle='->', lw=3, color='#27ae60'))
    ax.text(6.95, 4.8, 'Page In', fontsize=9, ha='center', color='#27ae60', weight='bold')

    # Timeline showing paging
    ax.text(7, 3.3, 'Paging Timeline During Training', fontsize=11, weight='bold', ha='center')

    timeline_data = [
        (1, 'Layer 1 active\nOthers paged out', '#27ae60'),
        (3, 'Layer 2 active\nLayer 1 paged out', '#3498db'),
        (5, 'Layer 3 active\nLayer 2 paged out', '#f39c12'),
        (7, 'All layers\nprocessed', '#9b59b6')
    ]

    for x, label, color in timeline_data:
        circle = plt.Circle((x * 3.2, 2.5), 0.3, color=color, ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x * 3.2, 1.8, label, fontsize=8, ha='center', va='top')

    # Connect timeline
    for i in range(len(timeline_data) - 1):
        x1 = timeline_data[i][0] * 3.2 + 0.3
        x2 = timeline_data[i+1][0] * 3.2 - 0.3
        ax.plot([x1, x2], [2.5, 2.5], 'k-', linewidth=2, alpha=0.5)

    # Performance impact
    perf_box = FancyBboxPatch((0.5, 0.2), 13, 1, boxstyle="round,pad=0.1",
                              facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=2)
    ax.add_patch(perf_box)
    ax.text(7, 0.9, 'Performance Impact', fontsize=11, weight='bold', ha='center')
    ax.text(3.5, 0.5, '✓ Overhead: <5%', fontsize=9, ha='center', color='green')
    ax.text(7, 0.5, '✓ Prevents OOM errors', fontsize=9, ha='center', color='green')
    ax.text(10.5, 0.5, '✓ Transparent to user', fontsize=9, ha='center', color='green')

    plt.tight_layout()
    plt.savefig(output_dir / 'paged_optimizers.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: paged_optimizers.png")

def create_training_efficiency():
    """Training efficiency comparison"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Training speed comparison
    batch_sizes = [1, 2, 4, 8, 16]

    full_speed = [0.8, 1.5, 2.8, 4.5, 0]  # 0 = OOM
    lora_speed = [1.0, 2.0, 4.0, 7.5, 13]
    qlora_speed = [0.9, 1.8, 3.6, 7.0, 13.5]

    ax1.plot(batch_sizes[:4], full_speed[:4], 'o-', linewidth=2.5, markersize=8,
             label='Full Fine-Tune', color='#e74c3c')
    ax1.plot(batch_sizes, lora_speed, 's-', linewidth=2.5, markersize=8,
             label='LoRA (FP16)', color='#3498db')
    ax1.plot(batch_sizes, qlora_speed, '^-', linewidth=2.5, markersize=8,
             label='QLoRA (4-bit)', color='#27ae60')

    ax1.axvline(x=4, color='red', linestyle='--', alpha=0.5, label='Full FT OOM limit')

    ax1.set_xlabel('Batch Size', fontsize=11, weight='bold')
    ax1.set_ylabel('Throughput (samples/sec)', fontsize=11, weight='bold')
    ax1.set_title('Training Speed vs Batch Size (7B Model)', fontsize=12, weight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xticks(batch_sizes)

    # 2. Time to convergence
    methods = ['Full\nFine-Tune', 'LoRA', 'QLoRA']
    time_hours = [12, 8, 8.5]
    colors = ['#e74c3c', '#3498db', '#27ae60']

    bars = ax2.bar(methods, time_hours, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Time to Convergence (hours)', fontsize=11, weight='bold')
    ax2.set_title('Training Time (7B Model, Same Loss)', fontsize=12, weight='bold')
    ax2.grid(axis='y', alpha=0.3)

    for bar, time in zip(bars, time_hours):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{time}h', ha='center', va='bottom', fontsize=10, weight='bold')

    # 3. Accuracy vs memory trade-off
    memory = [56, 20, 5]  # GB
    accuracy = [100, 98.5, 97.2]  # Relative to full fine-tune
    method_labels = ['Full Fine-Tune', 'LoRA', 'QLoRA']

    scatter = ax3.scatter(memory, accuracy, s=[300, 300, 300],
                         c=['#e74c3c', '#3498db', '#27ae60'],
                         alpha=0.6, edgecolors='black', linewidth=2)

    for i, label in enumerate(method_labels):
        ax3.annotate(label, (memory[i], accuracy[i]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    ax3.set_xlabel('GPU Memory (GB)', fontsize=11, weight='bold')
    ax3.set_ylabel('Relative Accuracy (%)', fontsize=11, weight='bold')
    ax3.set_title('Accuracy vs Memory Trade-off (7B Model)', fontsize=12, weight='bold')
    ax3.grid(alpha=0.3)
    ax3.set_ylim(95, 101)
    ax3.axhline(y=98, color='orange', linestyle='--', alpha=0.5, label='Acceptable threshold')
    ax3.legend()

    # 4. Cost comparison
    model_sizes = ['7B', '13B', '30B', '65B']
    cost_full = [150, 300, 800, 2000]
    cost_lora = [80, 150, 400, 1000]
    cost_qlora = [20, 40, 100, 250]

    x = np.arange(len(model_sizes))
    width = 0.25

    ax4.bar(x - width, cost_full, width, label='Full Fine-Tune', color='#e74c3c', alpha=0.8)
    ax4.bar(x, cost_lora, width, label='LoRA', color='#3498db', alpha=0.8)
    ax4.bar(x + width, cost_qlora, width, label='QLoRA', color='#27ae60', alpha=0.8)

    ax4.set_ylabel('Training Cost ($)', fontsize=11, weight='bold')
    ax4.set_xlabel('Model Size', fontsize=11, weight='bold')
    ax4.set_title('Cloud Training Cost Comparison (1000 steps)', fontsize=12, weight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_sizes)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'training_efficiency.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: training_efficiency.png")

def main():
    """Generate all QLoRA visualizations"""
    print("=" * 60)
    print("Generating QLoRA Visualizations")
    print("=" * 60)

    create_qlora_architecture()
    create_quantization_comparison()
    create_double_quantization()
    create_memory_comparison()
    create_paged_optimizers()
    create_training_efficiency()

    print("=" * 60)
    print(f"✓ All visualizations saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
