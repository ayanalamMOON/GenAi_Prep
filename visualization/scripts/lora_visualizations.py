"""
LoRA (Low-Rank Adaptation) Visualization Generator
Generates all visualizations for the LoRA section of the LLM Study Material
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.patches import ConnectionPatch
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Output directory
OUTPUT_DIR = "../images/lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_lora_architecture_diagram():
    """Visualize LoRA architecture showing W + BA decomposition"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7, 7.5, 'LoRA Architecture: Low-Rank Adaptation',
            fontsize=16, fontweight='bold', ha='center')

    # Input
    input_box = FancyBboxPatch((0.5, 3.5), 1, 1, boxstyle="round,pad=0.1",
                                edgecolor='#2E86AB', facecolor='#A7C6DA',
                                linewidth=2, alpha=0.7)
    ax.add_patch(input_box)
    ax.text(1, 4, 'Input\nx', fontsize=11, ha='center', va='center', fontweight='bold')

    # Original path (W_0)
    ax.text(3.5, 6.5, 'Original Path (Frozen)', fontsize=12,
            fontweight='bold', ha='center', color='#666')

    w0_box = FancyBboxPatch((2.5, 3.3), 2, 1.4, boxstyle="round,pad=0.1",
                            edgecolor='#666', facecolor='#ddd',
                            linewidth=3, alpha=0.6)
    ax.add_patch(w0_box)
    ax.text(3.5, 4.3, 'Pre-trained\nWeights', fontsize=10, ha='center', va='center')
    ax.text(3.5, 3.8, r'$W_0 \in \mathbb{R}^{d \times k}$',
            fontsize=10, ha='center', va='center', style='italic')
    ax.text(3.5, 3.4, '(FROZEN ❄)', fontsize=9, ha='center', color='blue', fontweight='bold')

    # Arrow from input to W0
    arrow1 = FancyArrowPatch((1.5, 4), (2.5, 4), arrowstyle='->',
                            mutation_scale=20, linewidth=2, color='#666')
    ax.add_patch(arrow1)

    # LoRA path (B and A)
    ax.text(3.5, 1.8, 'LoRA Path (Trainable)', fontsize=12,
            fontweight='bold', ha='center', color='#06A77D')

    # A matrix
    a_box = FancyBboxPatch((2, 0.3), 1.5, 1, boxstyle="round,pad=0.1",
                           edgecolor='#06A77D', facecolor='#B5E2D4',
                           linewidth=2, alpha=0.8)
    ax.add_patch(a_box)
    ax.text(2.75, 0.9, 'Down-project', fontsize=9, ha='center', fontweight='bold')
    ax.text(2.75, 0.6, r'$A \in \mathbb{R}^{r \times k}$',
            fontsize=9, ha='center', style='italic')
    ax.text(2.75, 0.35, 'r << k', fontsize=8, ha='center', color='green')

    # B matrix
    b_box = FancyBboxPatch((4, 0.3), 1.5, 1, boxstyle="round,pad=0.1",
                           edgecolor='#06A77D', facecolor='#B5E2D4',
                           linewidth=2, alpha=0.8)
    ax.add_patch(b_box)
    ax.text(4.75, 0.9, 'Up-project', fontsize=9, ha='center', fontweight='bold')
    ax.text(4.75, 0.6, r'$B \in \mathbb{R}^{d \times r}$',
            fontsize=9, ha='center', style='italic')
    ax.text(4.75, 0.35, 'r << d', fontsize=8, ha='center', color='green')

    # Arrows for LoRA path
    arrow2 = FancyArrowPatch((1.5, 4), (1.5, 2), arrowstyle='->',
                            mutation_scale=20, linewidth=2, color='#06A77D',
                            linestyle='--')
    ax.add_patch(arrow2)
    arrow3 = FancyArrowPatch((1.5, 2), (2, 0.8), arrowstyle='->',
                            mutation_scale=20, linewidth=2, color='#06A77D')
    ax.add_patch(arrow3)
    ax.text(1.3, 3, 'x', fontsize=10, ha='center', style='italic', color='#06A77D')

    arrow4 = FancyArrowPatch((3.5, 0.8), (4, 0.8), arrowstyle='->',
                            mutation_scale=20, linewidth=2, color='#06A77D')
    ax.add_patch(arrow4)
    ax.text(3.75, 1.1, r'$Ax$', fontsize=9, ha='center', style='italic', color='#06A77D')

    # Rank annotation
    rank_box = Rectangle((2.3, 1.5), 3, 0.4, facecolor='yellow',
                         edgecolor='orange', linewidth=2, alpha=0.6)
    ax.add_patch(rank_box)
    ax.text(3.8, 1.7, r'Bottleneck: rank $r$ (e.g., 8, 16, 32)',
            fontsize=10, ha='center', fontweight='bold')

    # Addition operation
    add_circle = Circle((6.5, 4), 0.4, facecolor='#F77F00',
                       edgecolor='black', linewidth=2)
    ax.add_patch(add_circle)
    ax.text(6.5, 4, '+', fontsize=20, ha='center', va='center',
           fontweight='bold', color='white')

    # Arrows to addition
    arrow5 = FancyArrowPatch((4.5, 4), (6.1, 4), arrowstyle='->',
                            mutation_scale=20, linewidth=2.5, color='#666')
    ax.add_patch(arrow5)
    ax.text(5.3, 4.3, r'$W_0 x$', fontsize=10, ha='center', style='italic')

    arrow6 = FancyArrowPatch((5.5, 0.8), (6.3, 3.6), arrowstyle='->',
                            mutation_scale=20, linewidth=2.5, color='#06A77D')
    ax.add_patch(arrow6)
    ax.text(5.8, 2.2, r'$\frac{\alpha}{r} BAx$', fontsize=10, ha='center',
           style='italic', color='#06A77D')

    # Output
    output_box = FancyBboxPatch((7.5, 3.5), 1.2, 1, boxstyle="round,pad=0.1",
                                edgecolor='#D62828', facecolor='#F4A5A5',
                                linewidth=2, alpha=0.7)
    ax.add_patch(output_box)
    ax.text(8.1, 4.3, 'Output', fontsize=11, ha='center', fontweight='bold')
    ax.text(8.1, 3.8, r'$h$', fontsize=11, ha='center', style='italic')

    arrow7 = FancyArrowPatch((6.9, 4), (7.5, 4), arrowstyle='->',
                            mutation_scale=20, linewidth=2.5, color='#D62828')
    ax.add_patch(arrow7)

    # Formula annotation
    formula_box = FancyBboxPatch((9.5, 3), 4, 2, boxstyle="round,pad=0.15",
                                 edgecolor='black', facecolor='lightyellow',
                                 linewidth=2, alpha=0.9)
    ax.add_patch(formula_box)
    ax.text(11.5, 4.5, 'LoRA Forward Pass:', fontsize=12,
           ha='center', fontweight='bold')
    ax.text(11.5, 4, r'$h = W_0 x + \frac{\alpha}{r} BAx$',
           fontsize=13, ha='center', style='italic')
    ax.text(11.5, 3.4, r'where $\alpha$ = scaling factor',
           fontsize=9, ha='center')

    # Parameter count comparison
    param_box = FancyBboxPatch((9.5, 0.3), 4, 2.3, boxstyle="round,pad=0.15",
                               edgecolor='#06A77D', facecolor='#E8F5E9',
                               linewidth=2, alpha=0.9)
    ax.add_patch(param_box)
    ax.text(11.5, 2.3, 'Parameter Efficiency:', fontsize=12,
           ha='center', fontweight='bold', color='#06A77D')
    ax.text(11.5, 1.9, r'Full: $d \times k$ parameters',
           fontsize=10, ha='center')
    ax.text(11.5, 1.5, r'LoRA: $r(d + k)$ parameters',
           fontsize=10, ha='center', color='#06A77D', fontweight='bold')
    ax.text(11.5, 1.1, 'Example: d=768, k=768, r=8', fontsize=9, ha='center')
    ax.text(11.5, 0.7, 'Full: 589,824 | LoRA: 12,288', fontsize=9, ha='center')
    ax.text(11.5, 0.4, '✓ 98% reduction!', fontsize=10, ha='center',
           color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/lora_architecture.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: lora_architecture.png")
    plt.close()


def create_rank_comparison():
    """Compare parameter counts and performance with different ranks"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Rank values
    ranks = [1, 2, 4, 8, 16, 32, 64, 128]
    d, k = 768, 768  # GPT-2 hidden dimension

    # Calculate parameters
    full_params = d * k
    lora_params = [r * (d + k) for r in ranks]
    reduction = [(full_params - lp) / full_params * 100 for lp in lora_params]

    # Left plot: Parameter count
    ax1.plot(ranks, lora_params, marker='o', linewidth=3, markersize=10,
            color='#06A77D', label='LoRA Parameters')
    ax1.axhline(y=full_params, color='#D62828', linestyle='--', linewidth=2.5,
               label=f'Full Fine-tuning ({full_params:,})')
    ax1.fill_between(ranks, 0, lora_params, alpha=0.2, color='#06A77D')

    ax1.set_xlabel('Rank (r)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Trainable Parameters', fontsize=13, fontweight='bold')
    ax1.set_title('LoRA Parameter Count vs Rank', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')

    # Add annotations
    for r, p in zip([8, 32], [lora_params[ranks.index(8)], lora_params[ranks.index(32)]]):
        ax1.annotate(f'r={r}\n{p:,} params\n({reduction[ranks.index(r)]:.1f}% reduction)',
                    xy=(r, p), xytext=(r*1.5, p*3),
                    arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                    fontsize=9, ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Right plot: Performance vs rank
    # Simulated performance (typical pattern from papers)
    performance = [0.65, 0.75, 0.85, 0.93, 0.96, 0.97, 0.975, 0.98]  # % of full fine-tuning

    ax2.plot(ranks, performance, marker='s', linewidth=3, markersize=10,
            color='#2E86AB', label='Task Performance')
    ax2.axhline(y=1.0, color='#D62828', linestyle='--', linewidth=2.5,
               label='Full Fine-tuning (100%)')
    ax2.fill_between(ranks, 0, performance, alpha=0.2, color='#2E86AB')

    ax2.set_xlabel('Rank (r)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Performance (% of Full FT)', fontsize=13, fontweight='bold')
    ax2.set_title('Performance vs Rank Trade-off', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.set_ylim(0.5, 1.05)

    # Add sweet spot annotation
    ax2.annotate('Sweet spot:\nr=8-16\n93-96% performance\n<2% parameters',
                xy=(12, 0.945), xytext=(40, 0.75),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='green'),
                fontsize=11, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.suptitle('Rank Selection Analysis: Parameters vs Performance',
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rank_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: rank_comparison.png")
    plt.close()


def create_matrix_decomposition():
    """Visualize low-rank matrix decomposition"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(7, 6.5, 'Low-Rank Matrix Decomposition',
            fontsize=16, fontweight='bold', ha='center')

    # Full update matrix ΔW
    ax.text(2, 5.5, r'Full Update $\Delta W$', fontsize=12,
           fontweight='bold', ha='center')
    delta_w = Rectangle((0.5, 2.5), 3, 2.5, facecolor='#D62828',
                        edgecolor='black', linewidth=2, alpha=0.4)
    ax.add_patch(delta_w)
    ax.text(2, 3.7, r'$d \times k$', fontsize=11, ha='center', style='italic')
    ax.text(2, 3.2, '(768 × 768)', fontsize=10, ha='center')
    ax.text(2, 2.8, '589,824 params', fontsize=9, ha='center', fontweight='bold')

    # Equals sign
    ax.text(4, 3.7, '≈', fontsize=40, ha='center', va='center')

    # B matrix
    ax.text(6.5, 5.5, 'Low-Rank Decomposition', fontsize=12,
           fontweight='bold', ha='center', color='#06A77D')
    b_matrix = Rectangle((5.3, 2.5), 1, 2.5, facecolor='#06A77D',
                         edgecolor='black', linewidth=2, alpha=0.6)
    ax.add_patch(b_matrix)
    ax.text(5.8, 3.7, r'$B$', fontsize=14, ha='center', style='italic',
           fontweight='bold')
    ax.text(5.8, 3.2, r'$d \times r$', fontsize=10, ha='center', style='italic')
    ax.text(5.8, 2.9, '(768×8)', fontsize=9, ha='center')

    # Multiplication sign
    ax.text(6.8, 3.7, '×', fontsize=30, ha='center', va='center')

    # A matrix
    a_matrix = Rectangle((7.5, 3.7), 2.5, 1, facecolor='#06A77D',
                         edgecolor='black', linewidth=2, alpha=0.6)
    ax.add_patch(a_matrix)
    ax.text(8.75, 4.2, r'$A$', fontsize=14, ha='center', style='italic',
           fontweight='bold')
    ax.text(8.75, 4, r'$r \times k$', fontsize=10, ha='center', style='italic')
    ax.text(8.75, 3.85, '(8×768)', fontsize=9, ha='center')

    # Total parameters for LoRA
    total_box = FancyBboxPatch((5, 1.7), 5.5, 0.6, boxstyle="round,pad=0.1",
                               edgecolor='#06A77D', facecolor='#B5E2D4',
                               linewidth=2, alpha=0.8)
    ax.add_patch(total_box)
    ax.text(7.75, 2, r'Total LoRA params: $r(d + k) = 8(768 + 768) = 12,288$',
           fontsize=11, ha='center', fontweight='bold')

    # Reduction annotation
    reduction_box = FancyBboxPatch((5, 0.8), 5.5, 0.7, boxstyle="round,pad=0.1",
                                   edgecolor='green', facecolor='lightgreen',
                                   linewidth=2.5, alpha=0.8)
    ax.add_patch(reduction_box)
    ax.text(7.75, 1.15, r'✓ Reduction: $\frac{589,824 - 12,288}{589,824} = 97.9\%$',
           fontsize=12, ha='center', fontweight='bold', color='green')

    # Dimension annotations
    ax.annotate('', xy=(0.5, 5.2), xytext=(3.5, 5.2),
               arrowprops=dict(arrowstyle='<->', lw=2, color='blue'))
    ax.text(2, 5.4, 'k (768)', fontsize=10, ha='center', color='blue')

    ax.annotate('', xy=(0.3, 2.5), xytext=(0.3, 5),
               arrowprops=dict(arrowstyle='<->', lw=2, color='blue'))
    ax.text(0.1, 3.7, 'd\n(768)', fontsize=10, ha='center', va='center', color='blue')

    ax.annotate('', xy=(5.3, 5.2), xytext=(6.3, 5.2),
               arrowprops=dict(arrowstyle='<->', lw=2, color='green'))
    ax.text(5.8, 5.4, 'r (8)', fontsize=10, ha='center', color='green', fontweight='bold')

    # Intuition box
    intuition_box = FancyBboxPatch((10.8, 2), 3, 3.5, boxstyle="round,pad=0.15",
                                   edgecolor='#F77F00', facecolor='#FFF3E0',
                                   linewidth=2, alpha=0.9)
    ax.add_patch(intuition_box)
    ax.text(12.3, 5.2, 'Why does this work?', fontsize=12,
           ha='center', fontweight='bold', color='#F77F00')
    ax.text(12.3, 4.7, 'Intrinsic Dimensionality:', fontsize=10,
           ha='center', fontweight='bold')
    ax.text(12.3, 4.3, 'Task-specific adaptations', fontsize=9, ha='center')
    ax.text(12.3, 4, 'lie in a low-dimensional', fontsize=9, ha='center')
    ax.text(12.3, 3.7, 'subspace of the full', fontsize=9, ha='center')
    ax.text(12.3, 3.4, 'weight matrix.', fontsize=9, ha='center')
    ax.text(12.3, 2.9, 'We only need to learn', fontsize=9, ha='center')
    ax.text(12.3, 2.6, 'the directions (B)', fontsize=9, ha='center', style='italic')
    ax.text(12.3, 2.3, 'in this subspace!', fontsize=9, ha='center')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/matrix_decomposition.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: matrix_decomposition.png")
    plt.close()


def create_module_selection():
    """Visualize which modules to apply LoRA to"""
    fig, ax = plt.subplots(figsize=(12, 8))

    modules = ['Q\n(Query)', 'K\n(Key)', 'V\n(Value)', 'O\n(Output)',
               'FFN\nUp', 'FFN\nDown', 'Embed', 'LM\nHead']

    # Impact on performance (simulated data from papers)
    performance = [0.92, 0.89, 0.93, 0.85, 0.78, 0.75, 0.65, 0.70]

    # Parameter increase %
    param_increase = [2.5, 2.5, 2.5, 2.5, 4.0, 4.0, 1.5, 1.5]

    # Color code by recommendation
    colors = []
    for p, pi in zip(performance, param_increase):
        if p > 0.90:
            colors.append('#06A77D')  # Highly recommended (green)
        elif p > 0.85:
            colors.append('#F77F00')  # Recommended (orange)
        else:
            colors.append('#D62828')  # Not recommended (red)

    # Create grouped bar chart
    x = np.arange(len(modules))
    width = 0.35

    bars1 = ax.bar(x - width/2, performance, width, label='Performance (% of full)',
                   color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

    bars2 = ax.bar(x + width/2, [pi/5 for pi in param_increase], width,
                   label='Parameter Increase (scaled)',
                   color='#2E86AB', edgecolor='black', linewidth=1.5, alpha=0.6)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{param_increase[i]:.1f}%',
                ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Module', fontsize=13, fontweight='bold')
    ax.set_ylabel('Performance / Parameters', fontsize=13, fontweight='bold')
    ax.set_title('LoRA Module Selection: Performance vs Parameter Cost',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(modules, fontsize=10)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.1)

    # Add recommendation boxes
    ax.text(0.02, 0.98, '● Highly Recommended (Q, V)',
           transform=ax.transAxes, fontsize=11, va='top',
           color='#06A77D', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    ax.text(0.02, 0.92, '● Recommended (K, O)',
           transform=ax.transAxes, fontsize=11, va='top',
           color='#F77F00', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#FFE0B2', alpha=0.7))

    ax.text(0.02, 0.86, '● Optional (FFN, Embed, Head)',
           transform=ax.transAxes, fontsize=11, va='top',
           color='#D62828', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.7))

    # Best practice annotation
    ax.annotate('Best Practice:\nApply to Q, V only\n(Good performance,\nlow parameters)',
                xy=(1.5, 0.925), xytext=(4, 0.75),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='green'),
                fontsize=10, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/module_selection.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: module_selection.png")
    plt.close()


def create_adapter_merging():
    """Visualize LoRA adapter merging and swapping"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7, 7.5, 'LoRA Adapter Modularity: Merge & Swap',
            fontsize=16, fontweight='bold', ha='center')

    # Base model (center)
    base_box = FancyBboxPatch((5.5, 3.5), 3, 1.5, boxstyle="round,pad=0.15",
                              edgecolor='#2E86AB', facecolor='#A7C6DA',
                              linewidth=3, alpha=0.7)
    ax.add_patch(base_box)
    ax.text(7, 4.5, 'Base Model', fontsize=13, ha='center', fontweight='bold')
    ax.text(7, 4.1, '(Frozen Weights)', fontsize=10, ha='center')
    ax.text(7, 3.7, 'GPT-2 (500MB)', fontsize=9, ha='center', style='italic')

    # Adapter 1 - Medical
    adapter1_box = FancyBboxPatch((1, 5.5), 2, 1.2, boxstyle="round,pad=0.1",
                                  edgecolor='#06A77D', facecolor='#B5E2D4',
                                  linewidth=2, alpha=0.8)
    ax.add_patch(adapter1_box)
    ax.text(2, 6.4, 'Adapter: Medical', fontsize=11, ha='center', fontweight='bold')
    ax.text(2, 6, 'LoRA weights', fontsize=9, ha='center')
    ax.text(2, 5.7, '(10MB)', fontsize=8, ha='center', style='italic')

    # Adapter 2 - Legal
    adapter2_box = FancyBboxPatch((1, 2), 2, 1.2, boxstyle="round,pad=0.1",
                                  edgecolor='#06A77D', facecolor='#B5E2D4',
                                  linewidth=2, alpha=0.8)
    ax.add_patch(adapter2_box)
    ax.text(2, 2.9, 'Adapter: Legal', fontsize=11, ha='center', fontweight='bold')
    ax.text(2, 2.5, 'LoRA weights', fontsize=9, ha='center')
    ax.text(2, 2.2, '(10MB)', fontsize=8, ha='center', style='italic')

    # Adapter 3 - Code
    adapter3_box = FancyBboxPatch((11, 3.5), 2, 1.2, boxstyle="round,pad=0.1",
                                  edgecolor='#06A77D', facecolor='#B5E2D4',
                                  linewidth=2, alpha=0.8)
    ax.add_patch(adapter3_box)
    ax.text(12, 4.4, 'Adapter: Code', fontsize=11, ha='center', fontweight='bold')
    ax.text(12, 4, 'LoRA weights', fontsize=9, ha='center')
    ax.text(12, 3.7, '(10MB)', fontsize=8, ha='center', style='italic')

    # Arrows showing swapping
    arrow1 = FancyArrowPatch((3, 6.1), (5.5, 4.8), arrowstyle='<->',
                            mutation_scale=25, linewidth=2.5, color='#F77F00')
    ax.add_patch(arrow1)
    ax.text(4, 5.6, 'Swap', fontsize=10, ha='center',
           fontweight='bold', color='#F77F00')

    arrow2 = FancyArrowPatch((3, 2.6), (5.5, 3.8), arrowstyle='<->',
                            mutation_scale=25, linewidth=2.5, color='#F77F00')
    ax.add_patch(arrow2)
    ax.text(4, 3, 'Swap', fontsize=10, ha='center',
           fontweight='bold', color='#F77F00')

    arrow3 = FancyArrowPatch((8.5, 4.2), (11, 4.2), arrowstyle='<->',
                            mutation_scale=25, linewidth=2.5, color='#F77F00')
    ax.add_patch(arrow3)
    ax.text(9.75, 4.5, 'Swap', fontsize=10, ha='center',
           fontweight='bold', color='#F77F00')

    # Merging option
    merge_box = FancyBboxPatch((4.5, 0.5), 5, 1.2, boxstyle="round,pad=0.15",
                               edgecolor='#D62828', facecolor='#F4A5A5',
                               linewidth=2.5, alpha=0.7)
    ax.add_patch(merge_box)
    ax.text(7, 1.4, 'Option: Merge Adapter into Base', fontsize=12,
           ha='center', fontweight='bold')
    ax.text(7, 0.95, r'$W_{new} = W_0 + \frac{\alpha}{r} BA$',
           fontsize=11, ha='center', style='italic')
    ax.text(7, 0.6, '(Creates new standalone model)', fontsize=9, ha='center')

    # Benefits annotation
    benefits_box = Rectangle((0.3, 0.3), 3.5, 1.5, facecolor='lightyellow',
                             edgecolor='orange', linewidth=2, alpha=0.8)
    ax.add_patch(benefits_box)
    ax.text(2.05, 1.6, 'Benefits:', fontsize=11, ha='center', fontweight='bold')
    ax.text(2.05, 1.3, '• One base model', fontsize=9, ha='center')
    ax.text(2.05, 1.05, '• Multiple tasks', fontsize=9, ha='center')
    ax.text(2.05, 0.8, '• Fast switching (<1s)', fontsize=9, ha='center')
    ax.text(2.05, 0.55, '• Minimal storage', fontsize=9, ha='center')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/adapter_merging.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: adapter_merging.png")
    plt.close()


def create_training_memory_breakdown():
    """Compare memory usage during training: Full FT vs LoRA"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Memory components
    components = ['Model\nWeights', 'Gradients', 'Optimizer\nStates',
                  'Activations', 'Total']

    # Full Fine-tuning (GPT-2 124M params)
    full_ft = [0.5, 0.5, 2.0, 1.0, 4.0]  # GB

    # LoRA (only 294K trainable params)
    lora = [0.5, 0.001, 0.004, 0.5, 1.005]  # GB

    x = np.arange(len(components))
    width = 0.35

    # Left plot - Full Fine-tuning
    bars1 = ax1.bar(x, full_ft, width, color=['#2E86AB', '#06A77D', '#F77F00', '#9C27B0', '#D62828'],
                   edgecolor='black', linewidth=1.5, alpha=0.8)

    for bar, val in zip(bars1, full_ft):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f} GB',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylabel('Memory (GB)', fontsize=13, fontweight='bold')
    ax1.set_title('Full Fine-Tuning Memory\n(GPT-2 124M)',
                 fontsize=13, fontweight='bold', color='#D62828')
    ax1.set_xticks(x)
    ax1.set_xticklabels(components, fontsize=10)
    ax1.set_ylim(0, 4.5)
    ax1.grid(axis='y', alpha=0.3)

    # Right plot - LoRA
    bars2 = ax2.bar(x, lora, width, color=['#2E86AB', '#06A77D', '#F77F00', '#9C27B0', '#06A77D'],
                   edgecolor='black', linewidth=1.5, alpha=0.8)

    for bar, val in zip(bars2, lora):
        height = bar.get_height()
        if val < 0.01:
            label = f'{val*1000:.1f} MB'
        else:
            label = f'{val:.2f} GB'
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                label,
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_ylabel('Memory (GB)', fontsize=13, fontweight='bold')
    ax2.set_title('LoRA Memory\n(294K trainable params)',
                 fontsize=13, fontweight='bold', color='#06A77D')
    ax2.set_xticks(x)
    ax2.set_xticklabels(components, fontsize=10)
    ax2.set_ylim(0, 4.5)
    ax2.grid(axis='y', alpha=0.3)

    # Add savings annotation
    savings = (full_ft[-1] - lora[-1]) / full_ft[-1] * 100
    fig.text(0.5, 0.02, f'✓ Memory Savings: {savings:.1f}% | ' +
             f'Full FT: {full_ft[-1]:.2f} GB → LoRA: {lora[-1]:.2f} GB',
             ha='center', fontsize=13, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen',
                      edgecolor='green', linewidth=2, alpha=0.8))

    plt.suptitle('Training Memory Breakdown: Full Fine-Tuning vs LoRA',
                fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(f"{OUTPUT_DIR}/training_memory_breakdown.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: training_memory_breakdown.png")
    plt.close()


def create_scaling_factor_impact():
    """Visualize the impact of alpha/r scaling factor"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Different alpha values (with r=8)
    r = 8
    alphas = [1, 8, 16, 32, 64, 128]
    scaling_factors = [a/r for a in alphas]

    # Simulated training loss convergence
    epochs = np.arange(0, 21)

    for alpha, sf in zip(alphas, scaling_factors):
        if alpha == 1:
            # Too small - slow convergence
            loss = 3.0 * np.exp(-0.05 * epochs) + 1.5
            label = f'α={alpha} (α/r={sf:.2f}) - Too small'
            color = '#D62828'
            linestyle = '--'
        elif alpha == 16:
            # Optimal
            loss = 3.0 * np.exp(-0.15 * epochs) + 0.5
            label = f'α={alpha} (α/r={sf:.1f}) - Optimal ✓'
            color = '#06A77D'
            linestyle = '-'
            linewidth = 3
        elif alpha == 128:
            # Too large - unstable
            loss = 2.5 + 0.5 * np.sin(epochs/3) * (1 + epochs/20)
            label = f'α={alpha} (α/r={sf:.1f}) - Too large'
            color = '#F77F00'
            linestyle = '-.'
        else:
            continue

        lw = 3 if alpha == 16 else 2
        ax1.plot(epochs, loss, linewidth=lw, label=label,
                color=color, linestyle=linestyle if alpha != 16 else '-')

    ax1.set_xlabel('Training Epochs', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax1.set_title('Impact of Scaling Factor (α/r) on Convergence',
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 5)

    # Right plot: Recommended alpha values
    ranks = [4, 8, 16, 32, 64]
    recommended_alphas = [8, 16, 32, 64, 128]  # Typically 2*r

    ax2.plot(ranks, recommended_alphas, marker='o', linewidth=3,
            markersize=12, color='#06A77D', label='Recommended α = 2r')
    ax2.plot(ranks, ranks, marker='s', linewidth=2, markersize=8,
            color='#2E86AB', linestyle='--', alpha=0.6, label='α = r')
    ax2.plot(ranks, [4*r for r in ranks], marker='^', linewidth=2,
            markersize=8, color='#F77F00', linestyle='--', alpha=0.6, label='α = 4r')

    ax2.set_xlabel('Rank (r)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Alpha (α)', fontsize=13, fontweight='bold')
    ax2.set_title('Recommended Alpha Values by Rank',
                 fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=2)

    # Add annotation
    ax2.annotate('Rule of thumb:\nα = 2 × r\n(balances learning\nrate and stability)',
                xy=(16, 32), xytext=(32, 16),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=10, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.suptitle('LoRA Scaling Factor Analysis', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/scaling_factor_impact.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: scaling_factor_impact.png")
    plt.close()


def create_inference_speed_comparison():
    """Compare inference speed: Full model vs LoRA vs merged LoRA"""
    fig, ax = plt.subplots(figsize=(12, 7))

    methods = ['Full Fine-tuned\nModel', 'LoRA\n(Dynamic)', 'LoRA\n(Merged)']

    # Inference time (ms per token)
    inference_times = [12, 12.5, 12]  # LoRA dynamic has tiny overhead

    # Memory footprint (GB)
    memory = [0.5, 0.51, 0.5]  # LoRA adapter is tiny

    # Storage size (MB)
    storage = [500, 10, 500]  # Only adapters are small

    x = np.arange(len(methods))
    width = 0.25

    # Create grouped bars
    bars1 = ax.bar(x - width, inference_times, width, label='Inference Time (ms/token)',
                   color='#2E86AB', edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = ax.bar(x, [m*10 for m in memory], width, label='Memory (GB × 10)',
                   color='#06A77D', edgecolor='black', linewidth=1.5, alpha=0.8)
    bars3 = ax.bar(x + width, [s/50 for s in storage], width, label='Storage (MB / 50)',
                   color='#F77F00', edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels
    for bars, vals, suffix in [(bars1, inference_times, 'ms'),
                                (bars2, memory, 'GB'),
                                (bars3, storage, 'MB')]:
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}{suffix}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Value (scaled for visualization)', fontsize=13, fontweight='bold')
    ax.set_title('LoRA Inference & Storage Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add insight boxes
    ax.text(0.02, 0.95, '✓ Near-zero inference overhead',
           transform=ax.transAxes, fontsize=11, va='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax.text(0.02, 0.88, '✓ 50x storage savings (adapters only)',
           transform=ax.transAxes, fontsize=11, va='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax.text(0.02, 0.81, '✓ Merging eliminates overhead completely',
           transform=ax.transAxes, fontsize=11, va='top',
           bbox=dict(boxstyle='round', facecolor='#FFE0B2', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/inference_speed_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: inference_speed_comparison.png")
    plt.close()


def create_lora_variants():
    """Compare different LoRA variants (LoRA, AdaLoRA, QLoRA, etc.)"""
    fig, ax = plt.subplots(figsize=(12, 8))

    variants = ['Standard\nLoRA', 'AdaLoRA\n(Adaptive)', 'LoRA+\n(Improved)',
                'QLoRA\n(Quantized)', 'DoRA\n(Weight Decomp)']

    # Performance (% of full fine-tuning)
    performance = [0.95, 0.96, 0.97, 0.94, 0.96]

    # Memory efficiency (GB for 7B model)
    memory = [12, 13, 12, 6, 14]

    # Training time (hours for 1K samples)
    time = [2.0, 2.5, 2.0, 3.0, 2.3]

    x = np.arange(len(variants))

    # Create subplots within the figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))

    # Performance
    bars1 = ax1.barh(x, performance, color='#06A77D', edgecolor='black',
                     linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Performance (% of Full FT)', fontsize=12, fontweight='bold')
    ax1.set_title('Task Performance', fontsize=13, fontweight='bold')
    ax1.set_yticks(x)
    ax1.set_yticklabels(variants, fontsize=10)
    ax1.set_xlim(0.90, 1.0)
    ax1.grid(axis='x', alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars1, performance)):
        width = bar.get_width()
        ax1.text(width + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.2%}',
                ha='left', va='center', fontsize=10, fontweight='bold')

    # Memory
    bars2 = ax2.barh(x, memory, color='#2E86AB', edgecolor='black',
                     linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Memory (GB for 7B model)', fontsize=12, fontweight='bold')
    ax2.set_title('Memory Efficiency', fontsize=13, fontweight='bold')
    ax2.set_yticks(x)
    ax2.set_yticklabels(variants, fontsize=10)
    ax2.grid(axis='x', alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars2, memory)):
        width = bar.get_width()
        ax2.text(width + 0.2, bar.get_y() + bar.get_height()/2,
                f'{val} GB',
                ha='left', va='center', fontsize=10, fontweight='bold')

    # Color code by memory (lower is better)
    for i, bar in enumerate(bars2):
        if memory[i] < 8:
            bar.set_facecolor('#06A77D')
        elif memory[i] < 13:
            bar.set_facecolor('#2E86AB')
        else:
            bar.set_facecolor('#F77F00')

    # Training time
    bars3 = ax3.barh(x, time, color='#F77F00', edgecolor='black',
                     linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('Training Time (hours)', fontsize=12, fontweight='bold')
    ax3.set_title('Training Speed', fontsize=13, fontweight='bold')
    ax3.set_yticks(x)
    ax3.set_yticklabels(variants, fontsize=10)
    ax3.grid(axis='x', alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars3, time)):
        width = bar.get_width()
        ax3.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}h',
                ha='left', va='center', fontsize=10, fontweight='bold')

    plt.suptitle('LoRA Variants Comparison (7B Model, 1K Samples)',
                fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/lora_variants.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: lora_variants.png")
    plt.close()


def main():
    """Generate all visualizations"""
    print("=" * 60)
    print("Generating LoRA Visualizations")
    print("=" * 60)

    create_lora_architecture_diagram()
    create_rank_comparison()
    create_matrix_decomposition()
    create_module_selection()
    create_adapter_merging()
    create_training_memory_breakdown()
    create_scaling_factor_impact()
    create_inference_speed_comparison()
    create_lora_variants()

    print("=" * 60)
    print(f"✓ All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
