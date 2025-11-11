"""
Pre-Training Visualization Generator
Generates all visualizations for the Pre-Training section of the LLM Study Material
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Wedge
from matplotlib.patches import ConnectionPatch
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Output directory
OUTPUT_DIR = "../../images/pretraining"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_pretraining_pipeline():
    """Visualize the complete pre-training pipeline"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Title
    ax.text(8, 8.5, 'Pre-Training Pipeline: From Raw Data to Foundation Model',
            fontsize=16, fontweight='bold', ha='center')

    # Step 1: Raw Data
    data_box = FancyBboxPatch((0.5, 6), 2, 1.5, boxstyle="round,pad=0.1",
                              edgecolor='#2E86AB', facecolor='#A7C6DA',
                              linewidth=2, alpha=0.7)
    ax.add_patch(data_box)
    ax.text(1.5, 7.2, 'Raw Corpus', fontsize=12, ha='center', fontweight='bold')
    ax.text(1.5, 6.8, 'Books, Web,', fontsize=9, ha='center')
    ax.text(1.5, 6.5, 'Wikipedia', fontsize=9, ha='center')
    ax.text(1.5, 6.2, '(TB of text)', fontsize=8, ha='center', style='italic')

    # Arrow 1
    arrow1 = FancyArrowPatch((2.5, 6.75), (3.8, 6.75), arrowstyle='->',
                            mutation_scale=25, linewidth=2.5, color='black')
    ax.add_patch(arrow1)
    ax.text(3.15, 7, 'Tokenize', fontsize=9, ha='center', fontweight='bold')

    # Step 2: Tokenization
    token_box = FancyBboxPatch((3.8, 6), 2, 1.5, boxstyle="round,pad=0.1",
                               edgecolor='#06A77D', facecolor='#B5E2D4',
                               linewidth=2, alpha=0.7)
    ax.add_patch(token_box)
    ax.text(4.8, 7.2, 'Tokenization', fontsize=12, ha='center', fontweight='bold')
    ax.text(4.8, 6.8, 'BPE/WordPiece', fontsize=9, ha='center')
    ax.text(4.8, 6.5, 'Vocab: 50K', fontsize=9, ha='center')
    ax.text(4.8, 6.2, '[1, 234, 5678...]', fontsize=8, ha='center', style='italic')

    # Arrow 2
    arrow2 = FancyArrowPatch((5.8, 6.75), (7.1, 6.75), arrowstyle='->',
                            mutation_scale=25, linewidth=2.5, color='black')
    ax.add_patch(arrow2)
    ax.text(6.45, 7, 'Build', fontsize=9, ha='center', fontweight='bold')

    # Step 3: Model Architecture
    arch_box = FancyBboxPatch((7.1, 6), 2.5, 1.5, boxstyle="round,pad=0.1",
                              edgecolor='#F77F00', facecolor='#FFE0B2',
                              linewidth=2, alpha=0.7)
    ax.add_patch(arch_box)
    ax.text(8.35, 7.2, 'Model Init', fontsize=12, ha='center', fontweight='bold')
    ax.text(8.35, 6.8, 'Random Weights', fontsize=9, ha='center')
    ax.text(8.35, 6.5, 'Layers: 12', fontsize=9, ha='center')
    ax.text(8.35, 6.2, 'Params: 124M', fontsize=8, ha='center', style='italic')

    # Arrow 3
    arrow3 = FancyArrowPatch((9.6, 6.75), (10.9, 6.75), arrowstyle='->',
                            mutation_scale=25, linewidth=2.5, color='black')
    ax.add_patch(arrow3)
    ax.text(10.25, 7, 'Train', fontsize=9, ha='center', fontweight='bold')

    # Step 4: Training
    train_box = FancyBboxPatch((10.9, 6), 2.5, 1.5, boxstyle="round,pad=0.1",
                               edgecolor='#9C27B0', facecolor='#E1BEE7',
                               linewidth=2, alpha=0.7)
    ax.add_patch(train_box)
    ax.text(12.15, 7.2, 'Next Token', fontsize=12, ha='center', fontweight='bold')
    ax.text(12.15, 6.95, 'Prediction', fontsize=12, ha='center', fontweight='bold')
    ax.text(12.15, 6.5, 'Epochs: 1-3', fontsize=9, ha='center')
    ax.text(12.15, 6.2, 'Days to weeks', fontsize=8, ha='center', style='italic')

    # Arrow 4
    arrow4 = FancyArrowPatch((13.4, 6.75), (14.2, 6.75), arrowstyle='->',
                            mutation_scale=25, linewidth=2.5, color='black')
    ax.add_patch(arrow4)

    # Step 5: Pre-trained Model
    model_box = FancyBboxPatch((14.2, 6), 1.5, 1.5, boxstyle="round,pad=0.1",
                               edgecolor='#D62828', facecolor='#F4A5A5',
                               linewidth=3, alpha=0.7)
    ax.add_patch(model_box)
    ax.text(14.95, 7.2, 'Foundation', fontsize=11, ha='center', fontweight='bold')
    ax.text(14.95, 6.8, 'Model', fontsize=11, ha='center', fontweight='bold')
    ax.text(14.95, 6.3, '✓ Ready for', fontsize=8, ha='center')
    ax.text(14.95, 6.1, 'fine-tuning', fontsize=8, ha='center')

    # Training details box
    details_box = FancyBboxPatch((0.5, 3.5), 15.2, 2, boxstyle="round,pad=0.15",
                                 edgecolor='#666', facecolor='#f5f5f5',
                                 linewidth=2, alpha=0.9)
    ax.add_patch(details_box)

    ax.text(8, 5.2, 'Pre-Training Configuration', fontsize=13,
           ha='center', fontweight='bold', color='#333')

    # Left column
    ax.text(1.5, 4.7, '📊 Data:', fontsize=10, ha='left', fontweight='bold')
    ax.text(1.5, 4.4, '  • Size: 300B-1T tokens', fontsize=9, ha='left')
    ax.text(1.5, 4.1, '  • Source: Books, Web, Code', fontsize=9, ha='left')
    ax.text(1.5, 3.8, '  • Quality: Filtered & deduplicated', fontsize=9, ha='left')

    # Middle column
    ax.text(6, 4.7, '⚙️ Hyperparameters:', fontsize=10, ha='left', fontweight='bold')
    ax.text(6, 4.4, '  • Learning Rate: 3e-4 → 3e-5', fontsize=9, ha='left')
    ax.text(6, 4.1, '  • Batch Size: 512-4096', fontsize=9, ha='left')
    ax.text(6, 3.8, '  • Warmup: 2000 steps', fontsize=9, ha='left')

    # Right column
    ax.text(10.5, 4.7, '💻 Resources:', fontsize=10, ha='left', fontweight='bold')
    ax.text(10.5, 4.4, '  • GPUs: 8-1024 A100s', fontsize=9, ha='left')
    ax.text(10.5, 4.1, '  • Time: Days to weeks', fontsize=9, ha='left')
    ax.text(10.5, 3.8, '  • Cost: $100K-$10M', fontsize=9, ha='left')

    # Objective box
    obj_box = FancyBboxPatch((0.5, 0.5), 15.2, 2.5, boxstyle="round,pad=0.15",
                             edgecolor='#9C27B0', facecolor='#F3E5F5',
                             linewidth=2.5, alpha=0.9)
    ax.add_patch(obj_box)

    ax.text(8, 2.7, 'Training Objective: Next Token Prediction',
           fontsize=13, ha='center', fontweight='bold', color='#9C27B0')

    ax.text(8, 2.2, r'$\mathcal{L} = -\mathbb{E}_{x \sim \mathcal{D}} \left[\sum_{t=1}^{T} \log P_\theta(x_t | x_{<t})\right]$',
           fontsize=12, ha='center', style='italic')

    ax.text(8, 1.7, 'Given: "The cat sat on the"  →  Predict: "mat" (0.45), "floor" (0.23), "sofa" (0.18), ...',
           fontsize=10, ha='center', family='monospace')

    ax.text(8, 1.2, 'Model learns: grammar, facts, reasoning, common sense',
           fontsize=10, ha='center', style='italic')

    ax.text(8, 0.8, '✓ No labels needed - self-supervised learning from raw text',
           fontsize=10, ha='center', fontweight='bold', color='green')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pretraining_pipeline.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: pretraining_pipeline.png")
    plt.close()


def create_scaling_laws():
    """Visualize scaling laws for model, data, and compute"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

    # Model size scaling
    N = np.logspace(6, 11, 50)  # 1M to 100B parameters
    N_c = 8.8e13
    alpha_N = 0.076
    loss_N = (N_c / N) ** alpha_N

    ax1.loglog(N, loss_N, linewidth=3, color='#2E86AB', label='Scaling Law')
    ax1.scatter([1.24e8, 1.5e9, 6.7e9, 1.75e11], [3.5, 3.1, 2.8, 2.4],
               s=200, color='red', zorder=5, label='Actual Models')

    # Annotate models
    models = ['GPT-2\n(124M)', 'GPT-2 XL\n(1.5B)', 'GPT-3 6.7B', 'GPT-3 175B']
    positions = [(1.24e8, 3.5), (1.5e9, 3.1), (6.7e9, 2.8), (1.75e11, 2.4)]
    for model, pos in zip(models, positions):
        ax1.annotate(model, xy=pos, xytext=(pos[0]*2, pos[1]+0.15),
                    fontsize=8, ha='left',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax1.set_xlabel('Number of Parameters (N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Scaling Law: Model Size', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.text(0.05, 0.95, r'$\mathcal{L}(N) \propto N^{-0.076}$',
            transform=ax1.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Data size scaling
    D = np.logspace(7, 12, 50)  # 10M to 1T tokens
    D_c = 5e13
    alpha_D = 0.095
    loss_D = (D_c / D) ** alpha_D

    ax2.loglog(D, loss_D, linewidth=3, color='#06A77D', label='Scaling Law')
    ax2.scatter([1e9, 3e11, 3e11, 1.4e12], [3.6, 3.0, 2.9, 2.5],
               s=200, color='red', zorder=5, label='Actual Training')

    ax2.set_xlabel('Dataset Size (tokens)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Scaling Law: Dataset Size', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.text(0.05, 0.95, r'$\mathcal{L}(D) \propto D^{-0.095}$',
            transform=ax2.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Chinchilla scaling comparison
    compute_budgets = np.array([1, 2, 4, 8, 16, 32, 64])  # Relative compute

    # Gopher approach (more params, less data)
    gopher_params = compute_budgets ** 0.73
    gopher_tokens = compute_budgets ** 0.27

    # Chinchilla approach (balanced)
    chinchilla_params = compute_budgets ** 0.50
    chinchilla_tokens = compute_budgets ** 0.50

    ax3.plot(compute_budgets, gopher_params, 'o-', linewidth=3, markersize=10,
            color='#F77F00', label='Gopher Strategy (More Params)')
    ax3.plot(compute_budgets, chinchilla_params, 's-', linewidth=3, markersize=10,
            color='#06A77D', label='Chinchilla Strategy (Balanced)')

    ax3.set_xlabel('Relative Compute Budget', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Relative Model Size', fontsize=12, fontweight='bold')
    ax3.set_title('Chinchilla vs Gopher Scaling Strategy', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    ax3.set_yscale('log', base=2)

    # Add annotation
    ax3.annotate('Chinchilla:\n70B params, 1.4T tokens\nOutperforms Gopher\n(280B params, 300B tokens)',
                xy=(32, 32**0.5), xytext=(8, 20),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=9, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Data vs Params optimal ratio
    ax4.plot(compute_budgets, chinchilla_tokens, 'o-', linewidth=3, markersize=10,
            color='#9C27B0', label='Optimal Data')
    ax4.plot(compute_budgets, chinchilla_params, 's-', linewidth=3, markersize=10,
            color='#2E86AB', label='Optimal Params')
    ax4.fill_between(compute_budgets, chinchilla_params, chinchilla_tokens,
                     alpha=0.2, color='purple')

    ax4.set_xlabel('Relative Compute Budget', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Relative Size', fontsize=12, fontweight='bold')
    ax4.set_title('Compute-Optimal Scaling', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log', base=2)
    ax4.set_yscale('log', base=2)

    ax4.text(0.5, 0.05, r'Key Insight: $N_{opt} \propto C^{0.5}, \quad D_{opt} \propto C^{0.5}$',
            transform=ax4.transAxes, fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.suptitle('Pre-Training Scaling Laws: Model, Data, and Compute',
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/scaling_laws.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: scaling_laws.png")
    plt.close()


def create_training_dynamics():
    """Visualize training loss curves and convergence"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Training loss over time
    steps = np.linspace(0, 100000, 1000)

    # Different model sizes
    loss_small = 3.5 * np.exp(-steps/30000) + 2.8  # 124M
    loss_medium = 3.5 * np.exp(-steps/35000) + 2.5  # 1B
    loss_large = 3.5 * np.exp(-steps/40000) + 2.2  # 7B
    loss_xlarge = 3.5 * np.exp(-steps/45000) + 1.9  # 70B

    ax1.plot(steps, loss_small, linewidth=2.5, label='Small (124M params)', color='#F77F00')
    ax1.plot(steps, loss_medium, linewidth=2.5, label='Medium (1B params)', color='#2E86AB')
    ax1.plot(steps, loss_large, linewidth=2.5, label='Large (7B params)', color='#06A77D')
    ax1.plot(steps, loss_xlarge, linewidth=2.5, label='X-Large (70B params)', color='#9C27B0')

    ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Pre-Training Loss Curves by Model Size', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1.5, 7)

    # Add phases
    ax1.axvline(x=2000, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(2000, 6.5, 'Warmup\nEnds', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='pink', alpha=0.7))

    ax1.axvline(x=80000, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(80000, 6.5, 'Decay\nStarts', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Learning rate schedule
    steps_lr = np.linspace(0, 100000, 1000)
    warmup_steps = 2000

    # Warmup phase
    warmup_lr = 3e-4 * (steps_lr / warmup_steps)
    warmup_lr = np.where(steps_lr <= warmup_steps, warmup_lr, 3e-4)

    # Cosine decay after warmup
    decay_steps = steps_lr - warmup_steps
    cosine_lr = 3e-4 * 0.5 * (1 + np.cos(np.pi * decay_steps / (100000 - warmup_steps)))

    # Combine
    lr_schedule = np.where(steps_lr <= warmup_steps, warmup_lr, cosine_lr)
    lr_schedule = np.maximum(lr_schedule, 3e-5)  # Minimum LR

    ax2.plot(steps_lr, lr_schedule, linewidth=3, color='#D62828')
    ax2.fill_between(steps_lr, 0, lr_schedule, alpha=0.3, color='#D62828')

    ax2.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Learning Rate Schedule (Warmup + Cosine Decay)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 3.5e-4)

    # Annotate phases
    ax2.annotate('Linear Warmup\n(0 → 3e-4)',
                xy=(1000, 1.5e-4), xytext=(10000, 2.5e-4),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax2.annotate('Cosine Decay\n(3e-4 → 3e-5)',
                xy=(60000, 1.5e-4), xytext=(40000, 3e-4),
                arrowprops=dict(arrowstyle='->', lw=2, color='orange'),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='#FFE0B2', alpha=0.8))

    ax2.axhline(y=3e-4, color='green', linestyle='--', linewidth=1.5, alpha=0.6)
    ax2.text(5000, 3.1e-4, 'Max LR: 3e-4', fontsize=9, color='green', fontweight='bold')

    ax2.axhline(y=3e-5, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    ax2.text(90000, 3.5e-5, 'Min LR: 3e-5', fontsize=9, color='red', fontweight='bold')

    plt.suptitle('Pre-Training Dynamics: Loss and Learning Rate',
                fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_dynamics.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: training_dynamics.png")
    plt.close()


def create_tokenization_comparison():
    """Compare different tokenization strategies"""
    fig, ax = plt.subplots(figsize=(12, 8))

    strategies = ['Character\nLevel', 'BPE\n(GPT-2)', 'WordPiece\n(BERT)',
                  'SentencePiece\n(T5)', 'Unigram\n(XLNet)']

    # Metrics
    vocab_sizes = [256, 50257, 30522, 32000, 32000]
    tokens_per_word = [4.5, 1.3, 1.4, 1.2, 1.2]
    unk_rate = [0, 0.001, 0.005, 0.001, 0.002]
    training_speed = [0.5, 1.0, 0.95, 1.05, 0.9]  # Relative

    x = np.arange(len(strategies))
    width = 0.2

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Vocabulary size
    bars1 = ax1.bar(x, vocab_sizes, color='#2E86AB', edgecolor='black',
                    linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('Vocabulary Size', fontsize=12, fontweight='bold')
    ax1.set_title('Vocabulary Size Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=10)
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars1, vocab_sizes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Tokens per word
    bars2 = ax2.bar(x, tokens_per_word, color='#06A77D', edgecolor='black',
                    linewidth=1.5, alpha=0.8)
    ax2.set_ylabel('Avg Tokens per Word', fontsize=12, fontweight='bold')
    ax2.set_title('Tokenization Efficiency', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 5)

    for bar, val in zip(bars2, tokens_per_word):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add optimal range
    ax2.axhspan(1.0, 1.5, alpha=0.2, color='green')
    ax2.text(0.02, 0.95, 'Optimal Range', transform=ax2.transAxes,
            fontsize=9, va='top', color='green', fontweight='bold')

    # Unknown token rate
    bars3 = ax3.bar(x, [r*100 for r in unk_rate], color='#F77F00',
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    ax3.set_ylabel('Unknown Token Rate (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Handling of Unknown Words', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies, fontsize=10)
    ax3.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars3, unk_rate):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val*100:.2f}%' if val > 0 else '0%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Training speed
    bars4 = ax4.bar(x, training_speed, color='#9C27B0', edgecolor='black',
                    linewidth=1.5, alpha=0.8)
    ax4.set_ylabel('Relative Training Speed', fontsize=12, fontweight='bold')
    ax4.set_title('Training Performance', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(strategies, fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.6)
    ax4.text(0.5, 1.02, 'Baseline (BPE)', fontsize=9, color='red')

    for bar, val in zip(bars4, training_speed):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}x',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Tokenization Strategy Comparison', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tokenization_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: tokenization_comparison.png")
    plt.close()


def create_data_mixture():
    """Visualize optimal data mixture for pre-training"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Data sources pie chart
    sources = ['Web Pages\n(CommonCrawl)', 'Books', 'Wikipedia',
               'Code\n(GitHub)', 'Scientific\nPapers', 'News', 'Other']
    sizes = [60, 15, 5, 10, 5, 3, 2]
    colors = ['#2E86AB', '#06A77D', '#F77F00', '#9C27B0', '#D62828', '#FFB6C1', '#A9A9A9']
    explode = (0.05, 0, 0, 0.03, 0, 0, 0)

    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=sources,
                                        colors=colors, autopct='%1.1f%%',
                                        startangle=90, textprops={'fontsize': 10})

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    ax1.set_title('Typical Pre-Training Data Mixture\n(GPT-3 Style)',
                 fontsize=13, fontweight='bold', pad=20)

    # Quality vs quantity trade-off
    quality = np.array([9, 8, 9.5, 7, 9, 7.5, 6])
    quantity = np.array([95, 8, 2, 15, 3, 5, 4])  # Billions of tokens

    ax2.scatter(quantity, quality, s=[s*30 for s in sizes],
               c=colors, alpha=0.7, edgecolors='black', linewidth=2)

    # Annotate each point
    for i, source in enumerate(['Web', 'Books', 'Wiki', 'Code', 'Papers', 'News', 'Other']):
        ax2.annotate(source,
                    xy=(quantity[i], quality[i]),
                    xytext=(quantity[i]+2, quality[i]+0.2),
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.7))

    ax2.set_xlabel('Available Data (Billions of tokens)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Data Quality Score (0-10)', fontsize=12, fontweight='bold')
    ax2.set_title('Data Quality vs Availability Trade-off', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-5, 105)
    ax2.set_ylim(5, 10)

    # Add optimal zone
    ax2.axhspan(8, 10, alpha=0.1, color='green')
    ax2.text(0.98, 0.95, 'High Quality\nZone', transform=ax2.transAxes,
            fontsize=10, va='top', ha='right', color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.suptitle('Pre-Training Data Composition and Quality',
                fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/data_mixture.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: data_mixture.png")
    plt.close()


def create_compute_requirements():
    """Visualize computational requirements for different model sizes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    models = ['GPT-2\n124M', 'GPT-2 XL\n1.5B', 'GPT-3 6.7B', 'GPT-3 13B',
              'GPT-3 175B', 'LLaMA-7B', 'LLaMA-65B']
    params = [0.124, 1.5, 6.7, 13, 175, 7, 65]  # Billions

    # GPU days (estimated)
    gpu_days_a100 = [2, 15, 80, 150, 3500, 90, 1000]

    # Cost estimation ($1/hour for A100)
    costs = [d * 24 * 1 for d in gpu_days_a100]  # Dollars

    # Left plot - Training time
    colors_models = ['#2E86AB', '#2E86AB', '#06A77D', '#06A77D',
                     '#06A77D', '#F77F00', '#F77F00']

    bars1 = ax1.barh(models, gpu_days_a100, color=colors_models,
                     edgecolor='black', linewidth=1.5, alpha=0.8)

    ax1.set_xlabel('GPU-Days (A100)', fontsize=12, fontweight='bold')
    ax1.set_title('Training Time Requirements', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_xscale('log')

    for i, (bar, val) in enumerate(zip(bars1, gpu_days_a100)):
        width = bar.get_width()
        ax1.text(width * 1.2, bar.get_y() + bar.get_height()/2,
                f'{val:.0f} days',
                ha='left', va='center', fontsize=9, fontweight='bold')

    # Add time ranges
    ax1.axvline(x=7, color='green', linestyle='--', linewidth=2, alpha=0.6)
    ax1.text(7, 0.5, '1 week', fontsize=9, ha='center', color='green',
            rotation=90, va='bottom')

    ax1.axvline(x=30, color='orange', linestyle='--', linewidth=2, alpha=0.6)
    ax1.text(30, 0.5, '1 month', fontsize=9, ha='center', color='orange',
            rotation=90, va='bottom')

    # Right plot - Cost
    bars2 = ax2.barh(models, [c/1000 for c in costs], color=colors_models,
                     edgecolor='black', linewidth=1.5, alpha=0.8)

    ax2.set_xlabel('Training Cost ($1000s)', fontsize=12, fontweight='bold')
    ax2.set_title('Estimated Training Cost (A100 @ $1/hr)', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_xscale('log')

    for i, (bar, val) in enumerate(zip(bars2, costs)):
        width = bar.get_width()
        if val >= 1000000:
            label = f'${val/1000000:.1f}M'
        elif val >= 1000:
            label = f'${val/1000:.0f}K'
        else:
            label = f'${val:.0f}'

        ax2.text(width * 1.2, bar.get_y() + bar.get_height()/2,
                label,
                ha='left', va='center', fontsize=9, fontweight='bold')

    # Add cost thresholds
    ax2.axvline(x=10, color='green', linestyle='--', linewidth=2, alpha=0.6)
    ax2.text(10, 0.5, 'Academic\nBudget', fontsize=8, ha='center', color='green',
            rotation=90, va='bottom')

    ax2.axvline(x=1000, color='red', linestyle='--', linewidth=2, alpha=0.6)
    ax2.text(1000, 0.5, 'Enterprise\nOnly', fontsize=8, ha='center', color='red',
            rotation=90, va='bottom')

    plt.suptitle('Pre-Training Computational Requirements',
                fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/compute_requirements.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: compute_requirements.png")
    plt.close()


def create_pretraining_vs_finetuning():
    """Compare pre-training and fine-tuning characteristics"""
    fig, ax = plt.subplots(figsize=(14, 8))

    categories = ['Data\nSize', 'Training\nTime', 'Compute\nCost',
                  'Learning\nRate', 'GPU\nMemory', 'Expertise\nNeeded']

    pretraining = [100, 90, 95, 80, 85, 90]  # Normalized to 100
    finetuning = [5, 10, 8, 20, 25, 30]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, pretraining, width, label='Pre-Training',
                   color='#D62828', edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = ax.bar(x + width/2, finetuning, width, label='Fine-Tuning',
                   color='#06A77D', edgecolor='black', linewidth=1.5, alpha=0.8)

    ax.set_ylabel('Relative Resource Requirement', fontsize=13, fontweight='bold')
    ax.set_title('Pre-Training vs Fine-Tuning: Resource Comparison',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 110)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add comparison annotations
    comparisons = [
        'Pre-training: TB\nFine-tuning: GB',
        'Pre-training: Weeks\nFine-tuning: Hours',
        'Pre-training: $100K+\nFine-tuning: $100',
        'Pre-training: 3e-4\nFine-tuning: 3e-5',
        'Pre-training: 8-1024 GPUs\nFine-tuning: 1-8 GPUs',
        'Pre-training: ML Research\nFine-tuning: Applied ML'
    ]

    for i, (cat, comp) in enumerate(zip(categories, comparisons)):
        ax.text(i, -15, comp, fontsize=8, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    # Add summary box
    summary_text = (
        'Key Differences:\n'
        '✓ Pre-training: From scratch, general knowledge\n'
        '✓ Fine-tuning: Task adaptation, leverages transfer learning\n'
        '✓ Most practitioners use fine-tuning (95%)\n'
        '✓ Pre-training mainly for foundation model developers'
    )

    ax.text(0.98, 0.98, summary_text, transform=ax.transAxes,
           fontsize=10, va='top', ha='right',
           bbox=dict(boxstyle='round', facecolor='#E3F2FD',
                    edgecolor='#2E86AB', linewidth=2, alpha=0.9))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pretraining_vs_finetuning.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: pretraining_vs_finetuning.png")
    plt.close()


def create_optimizer_comparison():
    """Compare different optimizers for pre-training"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Loss convergence comparison
    steps = np.linspace(0, 10000, 500)

    # Simulated convergence curves
    sgd_loss = 4.0 * np.exp(-steps/8000) + 2.5 + 0.1 * np.sin(steps/200)
    adam_loss = 4.0 * np.exp(-steps/5000) + 2.1
    adamw_loss = 4.0 * np.exp(-steps/4500) + 2.0

    ax1.plot(steps, sgd_loss, linewidth=2.5, label='SGD', color='#F77F00')
    ax1.plot(steps, adam_loss, linewidth=2.5, label='Adam', color='#2E86AB')
    ax1.plot(steps, adamw_loss, linewidth=2.5, label='AdamW (Best)', color='#06A77D', linestyle='-')

    ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Optimizer Convergence Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1.8, 7)

    # Add annotation
    ax1.annotate('AdamW converges fastest\nand to lowest loss',
                xy=(8000, 2.0), xytext=(5000, 3.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=10, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Optimizer characteristics radar chart
    optimizers = ['AdamW', 'Adam', 'SGD', 'RMSprop']

    # Metrics (0-10 scale)
    metrics = {
        'Convergence Speed': [9, 8, 5, 7],
        'Final Performance': [9, 8, 7, 7],
        'Memory Efficiency': [7, 7, 10, 8],
        'Stability': [9, 7, 8, 7],
        'Hyperparameter Sensitivity': [8, 7, 5, 6]
    }

    # Create table instead of radar (simpler)
    metric_names = list(metrics.keys())
    data = []
    for opt in optimizers:
        row = [metrics[m][optimizers.index(opt)] for m in metric_names]
        data.append(row)

    # Heatmap
    im = ax2.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=10)

    ax2.set_xticks(np.arange(len(metric_names)))
    ax2.set_yticks(np.arange(len(optimizers)))
    ax2.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=10)
    ax2.set_yticklabels(optimizers, fontsize=11)
    ax2.set_title('Optimizer Characteristics Comparison', fontsize=13, fontweight='bold')

    # Add values
    for i in range(len(optimizers)):
        for j in range(len(metric_names)):
            text = ax2.text(j, i, f'{data[i][j]}/10',
                           ha="center", va="center", color="black",
                           fontweight='bold', fontsize=9)

    plt.colorbar(im, ax=ax2, label='Score (0-10)')

    plt.suptitle('Pre-Training Optimizer Analysis', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/optimizer_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: optimizer_comparison.png")
    plt.close()


def main():
    """Generate all visualizations"""
    print("=" * 60)
    print("Generating Pre-Training Visualizations")
    print("=" * 60)

    create_pretraining_pipeline()
    create_scaling_laws()
    create_training_dynamics()
    create_tokenization_comparison()
    create_data_mixture()
    create_compute_requirements()
    create_pretraining_vs_finetuning()
    create_optimizer_comparison()

    print("=" * 60)
    print(f"✓ All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
