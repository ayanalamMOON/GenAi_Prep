"""
Fine-Tuning Visualization Generator
Generates all visualizations for the Fine-Tuning section of the LLM Study Material
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Output directory
OUTPUT_DIR = "../images/fine_tuning"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_transfer_learning_diagram():
    """Visualize the transfer learning process: Pre-training -> Fine-tuning"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Pre-training phase
    pretrain_box = FancyBboxPatch((0.5, 3.5), 2.5, 1.8,
                                   boxstyle="round,pad=0.1",
                                   edgecolor='#2E86AB', facecolor='#A7C6DA',
                                   linewidth=2, alpha=0.7)
    ax.add_patch(pretrain_box)
    ax.text(1.75, 4.8, 'Pre-training', fontsize=14, fontweight='bold', ha='center')
    ax.text(1.75, 4.3, 'Large Corpus\n(billions of tokens)', fontsize=10, ha='center')
    ax.text(1.75, 3.8, r'$\theta_{pre-trained}$', fontsize=12, ha='center', style='italic')

    # Arrow
    arrow1 = FancyArrowPatch((3, 4.4), (4.5, 4.4),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color='#F77F00')
    ax.add_patch(arrow1)
    ax.text(3.75, 4.8, 'Transfer', fontsize=11, ha='center', fontweight='bold')

    # Fine-tuning phase
    finetune_box = FancyBboxPatch((4.5, 3.5), 2.5, 1.8,
                                   boxstyle="round,pad=0.1",
                                   edgecolor='#06A77D', facecolor='#B5E2D4',
                                   linewidth=2, alpha=0.7)
    ax.add_patch(finetune_box)
    ax.text(5.75, 4.8, 'Fine-tuning', fontsize=14, fontweight='bold', ha='center')
    ax.text(5.75, 4.3, 'Task-specific\nDataset', fontsize=10, ha='center')
    ax.text(5.75, 3.8, r'$\theta_{pre} + \Delta\theta$', fontsize=12, ha='center', style='italic')

    # Arrow to final model
    arrow2 = FancyArrowPatch((7, 4.4), (8.5, 4.4),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color='#F77F00')
    ax.add_patch(arrow2)
    ax.text(7.75, 4.8, 'Adapt', fontsize=11, ha='center', fontweight='bold')

    # Final model
    final_box = FancyBboxPatch((8.5, 3.5), 1.2, 1.8,
                               boxstyle="round,pad=0.05",
                               edgecolor='#D62828', facecolor='#F4A5A5',
                               linewidth=2, alpha=0.7)
    ax.add_patch(final_box)
    ax.text(9.1, 4.8, 'Adapted\nModel', fontsize=11, fontweight='bold', ha='center')
    ax.text(9.1, 3.9, r'$\theta_{fine}$', fontsize=11, ha='center', style='italic')

    # Bottom annotations
    ax.text(1.75, 2.8, 'All parameters\ninitialized', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(5.75, 2.8, 'All parameters\nupdated', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax.text(9.1, 2.8, 'Task-specific\nweights', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    plt.title('Transfer Learning in Fine-Tuning', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/transfer_learning_process.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: transfer_learning_process.png")
    plt.close()


def create_learning_rate_comparison():
    """Compare learning rates for pre-training vs fine-tuning"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = np.linspace(0, 100, 1000)

    # Pre-training LR schedule
    lr_pretrain_warmup = 3e-4 * (epochs / 10)
    lr_pretrain_warmup[epochs > 10] = 3e-4
    lr_pretrain_decay = 3e-4 * np.exp(-0.02 * (epochs - 10))
    lr_pretrain_decay[epochs <= 10] = lr_pretrain_warmup[epochs <= 10]

    ax1.plot(epochs, lr_pretrain_decay, linewidth=2.5, color='#2E86AB', label='Pre-training LR')
    ax1.axhline(y=3e-4, color='gray', linestyle='--', alpha=0.5, label='Peak LR = 3e-4')
    ax1.fill_between(epochs[:100], 0, lr_pretrain_decay[:100], alpha=0.2, color='#2E86AB')
    ax1.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax1.set_title('Pre-training: High Initial LR', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Fine-tuning LR schedule
    lr_finetune_warmup = 3e-5 * (epochs / 5)
    lr_finetune_warmup[epochs > 5] = 3e-5
    lr_finetune_decay = 3e-5 * np.exp(-0.05 * (epochs - 5))
    lr_finetune_decay[epochs <= 5] = lr_finetune_warmup[epochs <= 5]

    ax2.plot(epochs, lr_finetune_decay, linewidth=2.5, color='#06A77D', label='Fine-tuning LR')
    ax2.axhline(y=3e-5, color='gray', linestyle='--', alpha=0.5, label='Peak LR = 3e-5')
    ax2.fill_between(epochs[:100], 0, lr_finetune_decay[:100], alpha=0.2, color='#06A77D')
    ax2.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Fine-tuning: Low LR (1/10 of pre-training)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.suptitle('Learning Rate Schedules: Pre-training vs Fine-tuning',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/learning_rate_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: learning_rate_comparison.png")
    plt.close()


def create_discriminative_lr_diagram():
    """Visualize discriminative learning rates across layers"""
    fig, ax = plt.subplots(figsize=(12, 7))

    layers = np.arange(1, 13)  # 12 layers for GPT-2
    base_lr = 3e-5
    decay_factor = 0.85

    # Calculate LR for each layer
    lrs = [base_lr * (decay_factor ** (12 - l)) for l in layers]

    # Create bar plot
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(layers)))
    bars = ax.bar(layers, lrs, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels on bars
    for i, (bar, lr) in enumerate(zip(bars, lrs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{lr:.2e}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Layer Number', fontsize=13, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=13, fontweight='bold')
    ax.set_title('Discriminative Learning Rates Across Layers\n' +
                 r'$\eta_l = \eta_0 \cdot \gamma^{L-l}$ where $\gamma=0.85$',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(layers)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_yscale('log')

    # Add annotations
    ax.annotate('Lower layers:\nGeneral features\n(smaller updates)',
                xy=(2, lrs[1]), xytext=(3, 5e-6),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax.annotate('Upper layers:\nTask-specific\n(larger updates)',
                xy=(11, lrs[10]), xytext=(9, 4e-5),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/discriminative_learning_rates.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: discriminative_learning_rates.png")
    plt.close()


def create_gradient_flow_analysis():
    """Visualize gradient magnitudes across layers"""
    fig, ax = plt.subplots(figsize=(12, 7))

    layers = np.arange(1, 13)
    r = 0.75  # decay ratio
    L = 12

    # Calculate gradient magnitudes
    grad_magnitudes = [r ** (L - l) for l in layers]
    grad_magnitudes = np.array(grad_magnitudes) * 100  # Scale for visualization

    # Create line plot with markers
    ax.plot(layers, grad_magnitudes, marker='o', markersize=10,
            linewidth=3, color='#D62828', label='Gradient Magnitude')
    ax.fill_between(layers, 0, grad_magnitudes, alpha=0.3, color='#D62828')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    ax.set_xlabel('Layer Number (1=input, 12=output)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'Gradient Magnitude $\|\nabla W_l\|$', fontsize=13, fontweight='bold')
    ax.set_title('Gradient Flow: Magnitudes Decrease in Earlier Layers\n' +
                 r'$\|\nabla W_l\| \approx r^{L-l}$ where $r=0.75$',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(layers)
    ax.legend(fontsize=11)

    # Highlight the ratio
    ax.annotate(f'Layer 1: {grad_magnitudes[0]:.1f}\nLayer 12: {grad_magnitudes[11]:.1f}\n' +
                f'Ratio: {grad_magnitudes[11]/grad_magnitudes[0]:.1f}x',
                xy=(6.5, 50), fontsize=11,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Add annotation for vanishing gradient
    ax.annotate('Vanishing gradient problem:\nEarlier layers get tiny updates',
                xy=(2, grad_magnitudes[1]), xytext=(3, 80),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/gradient_flow_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: gradient_flow_analysis.png")
    plt.close()


def create_catastrophic_forgetting_diagram():
    """Visualize catastrophic forgetting phenomenon"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    epochs = np.arange(0, 51)

    # Without regularization
    task_a_perf = 95 * np.exp(-0.15 * epochs)  # Original task performance drops
    task_b_perf = 100 * (1 - np.exp(-0.1 * epochs))  # New task improves

    ax1.plot(epochs, task_a_perf, linewidth=3, color='#2E86AB',
             label='Original Task Performance', marker='o', markersize=4, markevery=5)
    ax1.plot(epochs, task_b_perf, linewidth=3, color='#06A77D',
             label='New Task Performance', marker='s', markersize=4, markevery=5)
    ax1.fill_between(epochs, 0, task_a_perf, alpha=0.2, color='#2E86AB')
    ax1.fill_between(epochs, 0, task_b_perf, alpha=0.2, color='#06A77D')

    ax1.set_xlabel('Fine-tuning Epochs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Without Regularization\n(Catastrophic Forgetting)',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='center right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)

    # Add critical annotation
    ax1.annotate('Forgetting!', xy=(25, task_a_perf[25]), xytext=(30, 20),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                fontsize=11, fontweight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # With EWC regularization
    task_a_perf_ewc = 95 - 10 * (1 - np.exp(-0.05 * epochs))  # Slower drop
    task_b_perf_ewc = 90 * (1 - np.exp(-0.08 * epochs))  # Slightly slower improvement

    ax2.plot(epochs, task_a_perf_ewc, linewidth=3, color='#2E86AB',
             label='Original Task (Protected)', marker='o', markersize=4, markevery=5)
    ax2.plot(epochs, task_b_perf_ewc, linewidth=3, color='#06A77D',
             label='New Task Performance', marker='s', markersize=4, markevery=5)
    ax2.fill_between(epochs, 0, task_a_perf_ewc, alpha=0.2, color='#2E86AB')
    ax2.fill_between(epochs, 0, task_b_perf_ewc, alpha=0.2, color='#06A77D')

    ax2.set_xlabel('Fine-tuning Epochs', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
    ax2.set_title('With EWC Regularization\n(Knowledge Preserved)',
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='center right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    # Add annotation
    ax2.annotate('Knowledge\nRetained!', xy=(40, task_a_perf_ewc[40]), xytext=(25, 70),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=11, fontweight='bold', color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.suptitle('Catastrophic Forgetting: With vs Without Regularization',
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/catastrophic_forgetting.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: catastrophic_forgetting.png")
    plt.close()


def create_fisher_information_heatmap():
    """Visualize Fisher Information Matrix for important parameters"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Simulate Fisher Information for 12 layers x 8 parameter groups
    np.random.seed(42)
    layers = 12
    param_groups = 8

    # Create synthetic Fisher Information (higher for later layers)
    fisher_info = np.zeros((layers, param_groups))
    for i in range(layers):
        # Later layers have higher importance (more task-specific)
        importance_factor = (i + 1) / layers
        fisher_info[i, :] = np.random.gamma(2, importance_factor * 5, param_groups)

    # Create heatmap
    im = ax.imshow(fisher_info, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fisher Information (Parameter Importance)',
                   fontsize=12, fontweight='bold')

    # Set labels
    ax.set_xlabel('Parameter Groups', fontsize=12, fontweight='bold')
    ax.set_ylabel('Layer Number', fontsize=12, fontweight='bold')
    ax.set_title('Fisher Information Matrix\nHigher values = More important for original task',
                 fontsize=14, fontweight='bold', pad=15)

    # Set ticks
    ax.set_xticks(np.arange(param_groups))
    ax.set_yticks(np.arange(layers))
    ax.set_xticklabels([f'P{i+1}' for i in range(param_groups)])
    ax.set_yticklabels([f'L{i+1}' for i in range(layers)])

    # Add text annotations for some cells
    for i in range(layers):
        for j in range(param_groups):
            if fisher_info[i, j] > 8:  # Highlight high-importance params
                text = ax.text(j, i, '!', ha="center", va="center",
                              color="black", fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fisher_information_matrix.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: fisher_information_matrix.png")
    plt.close()


def create_training_loss_curves():
    """Visualize typical training loss curves"""
    fig, ax = plt.subplots(figsize=(12, 7))

    steps = np.arange(0, 1001)

    # Different scenarios
    # Good convergence
    loss_good = 3.5 * np.exp(-0.005 * steps) + 0.5 + np.random.normal(0, 0.05, len(steps))

    # Overfitting
    loss_train_overfit = 3.5 * np.exp(-0.006 * steps) + 0.3
    loss_val_overfit = 3.5 * np.exp(-0.004 * steps) + 0.8 + 0.001 * (steps - 400)
    loss_val_overfit[:400] = 3.5 * np.exp(-0.004 * steps[:400]) + 0.8

    # Too high LR
    loss_high_lr = 2.5 + 0.5 * np.sin(steps / 30) + np.random.normal(0, 0.3, len(steps))

    ax.plot(steps, loss_good, linewidth=2.5, color='#06A77D',
            label='Good: Steady Convergence', alpha=0.8)
    ax.plot(steps, loss_train_overfit, linewidth=2.5, color='#2E86AB',
            label='Overfitting: Training Loss', linestyle='--', alpha=0.8)
    ax.plot(steps, loss_val_overfit, linewidth=2.5, color='#D62828',
            label='Overfitting: Validation Loss', alpha=0.8)
    ax.plot(steps, loss_high_lr, linewidth=2.5, color='#F77F00',
            label='Bad: High LR (unstable)', alpha=0.6)

    ax.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title('Fine-Tuning Loss Curves: Different Scenarios',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 5)

    # Add annotations
    ax.annotate('Overfitting starts here', xy=(400, loss_val_overfit[400]),
                xytext=(550, 2.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax.annotate('Oscillating\n(unstable)', xy=(500, loss_high_lr[500]),
                xytext=(700, 3.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='orange'),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_loss_curves.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: training_loss_curves.png")
    plt.close()


def create_memory_comparison():
    """Compare memory requirements: Full Fine-tuning vs LoRA"""
    fig, ax = plt.subplots(figsize=(12, 7))

    models = ['GPT-2\n(124M)', 'GPT-2-M\n(355M)', 'GPT-2-L\n(774M)', 'LLaMA-7B\n(7B)']
    full_finetuning = [4, 8, 16, 32]  # GB
    lora = [2, 3.5, 6, 12]  # GB

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, full_finetuning, width, label='Full Fine-tuning',
                   color='#D62828', edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = ax.bar(x + width/2, lora, width, label='LoRA (Parameter-Efficient)',
                   color='#06A77D', edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f} GB',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Model Size', fontsize=13, fontweight='bold')
    ax.set_ylabel('GPU Memory Required (GB)', fontsize=13, fontweight='bold')
    ax.set_title('Memory Requirements: Full Fine-tuning vs LoRA',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add savings annotation
    savings = [(full_finetuning[i] - lora[i]) / full_finetuning[i] * 100
               for i in range(len(models))]
    avg_savings = np.mean(savings)

    ax.text(0.98, 0.95, f'Average Memory Savings: {avg_savings:.1f}%',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/memory_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: memory_comparison.png")
    plt.close()


def create_hyperparameter_heatmap():
    """Visualize recommended hyperparameters for different scenarios"""
    fig, ax = plt.subplots(figsize=(12, 8))

    scenarios = ['Small Data\n(<1K samples)', 'Medium Data\n(1K-10K)',
                 'Large Data\n(10K-100K)', 'Very Large\n(>100K)']
    params = ['Learning Rate\n(×10⁻⁵)', 'Batch Size', 'Epochs',
              'Warmup (%)', 'Weight Decay\n(×10⁻²)']

    # Hyperparameter recommendations (normalized for visualization)
    data = np.array([
        [5, 4, 20, 10, 0.1],   # Small data
        [3, 8, 10, 5, 0.5],    # Medium data
        [2, 16, 5, 3, 1.0],    # Large data
        [1, 32, 3, 2, 1.5]     # Very large data
    ]).T

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Relative Value', fontsize=12, fontweight='bold')

    # Set labels
    ax.set_xlabel('Data Size Scenario', fontsize=13, fontweight='bold')
    ax.set_ylabel('Hyperparameter', fontsize=13, fontweight='bold')
    ax.set_title('Recommended Hyperparameters by Dataset Size',
                 fontsize=14, fontweight='bold', pad=15)

    # Set ticks
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_yticks(np.arange(len(params)))
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.set_yticklabels(params, fontsize=10)

    # Add text annotations
    for i in range(len(params)):
        for j in range(len(scenarios)):
            text = ax.text(j, i, f'{data[i, j]:.1f}',
                          ha="center", va="center", color="black",
                          fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hyperparameter_recommendations.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: hyperparameter_recommendations.png")
    plt.close()


def create_convergence_analysis():
    """Visualize convergence with different learning rates"""
    fig, ax = plt.subplots(figsize=(12, 7))

    iterations = np.arange(0, 201)
    optimal_theta = 0  # Optimal parameter value

    # Different learning rates
    lr_too_low = 0.01
    lr_optimal = 0.1
    lr_too_high = 0.5
    lr_diverge = 1.5

    # Simulate convergence
    def converge(lr, iterations):
        theta = 10  # Initial value
        path = [theta]
        for _ in range(len(iterations) - 1):
            gradient = 2 * (theta - optimal_theta)  # Simple quadratic loss
            theta = theta - lr * gradient
            path.append(theta)
        return np.array(path)

    path_low = converge(lr_too_low, iterations)
    path_optimal = converge(lr_optimal, iterations)
    path_high = converge(lr_too_high, iterations)
    path_diverge = converge(lr_diverge, iterations)

    # Clip diverging path for visualization
    path_diverge = np.clip(path_diverge, -20, 20)

    ax.plot(iterations, np.abs(path_low), linewidth=2.5,
            label=f'Too Low (η={lr_too_low})', color='#2E86AB', linestyle='--')
    ax.plot(iterations, np.abs(path_optimal), linewidth=3,
            label=f'Optimal (η={lr_optimal})', color='#06A77D')
    ax.plot(iterations, np.abs(path_high), linewidth=2.5,
            label=f'Too High (η={lr_too_high})', color='#F77F00', linestyle='-.')
    ax.plot(iterations, np.abs(path_diverge), linewidth=2.5,
            label=f'Diverging (η={lr_diverge})', color='#D62828', alpha=0.7)

    ax.set_xlabel('Iterations', fontsize=13, fontweight='bold')
    ax.set_ylabel('Distance from Optimal |θ - θ*|', fontsize=13, fontweight='bold')
    ax.set_title('Convergence Analysis: Impact of Learning Rate',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 30)

    # Add annotations
    ax.annotate('Slow convergence', xy=(100, np.abs(path_low[100])),
                xytext=(120, 5),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    ax.annotate('Fast & stable', xy=(50, np.abs(path_optimal[50])),
                xytext=(80, 0.01),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/convergence_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: convergence_analysis.png")
    plt.close()


def main():
    """Generate all visualizations"""
    print("=" * 60)
    print("Generating Fine-Tuning Visualizations")
    print("=" * 60)

    create_transfer_learning_diagram()
    create_learning_rate_comparison()
    create_discriminative_lr_diagram()
    create_gradient_flow_analysis()
    create_catastrophic_forgetting_diagram()
    create_fisher_information_heatmap()
    create_training_loss_curves()
    create_memory_comparison()
    create_hyperparameter_heatmap()
    create_convergence_analysis()

    print("=" * 60)
    print(f"✓ All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
