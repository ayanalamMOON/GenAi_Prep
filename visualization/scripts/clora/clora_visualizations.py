"""
C-LoRA (Continuous LoRA) Visualization Script

This script generates publication-quality visualizations for the C-LoRA section,
demonstrating gradient-flow-based rank adaptation and per-layer rank selection.

Visualizations:
1. Gate Evolution During Training (2 panels)
   - Left: Gate values over time
   - Right: Effective rank convergence
   
2. Per-Layer Rank Adaptation (2 panels)
   - Left: Heatmap of gate values across layers
   - Right: Effective rank by layer depth
   
3. Sparsity vs Accuracy Trade-off (2 panels)
   - Left: Effect of lambda on effective rank
   - Right: Validation accuracy vs effective rank

Output: 3 PNG files at 300 DPI in ../../images/clora/
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# CRITICAL: Output directory is TWO levels up from scripts/clora/
OUTPUT_DIR = "../../images/clora"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# High-resolution publication quality
DPI = 300

def create_gate_evolution():
    """
    Visualization 1: Gate Evolution During Training
    
    Left panel: Individual gate values over training steps
    Right panel: Effective rank convergence
    
    Shows how gates differentiate from uniform 0.5 to sparse 0/1 distribution.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('C-LoRA Gate Evolution During Training', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Simulate training dynamics
    num_steps = 1000
    num_gates = 16
    steps = np.arange(num_steps)
    
    # Generate gate trajectories (different decay rates for different gates)
    gates = np.zeros((num_steps, num_gates))
    
    # Initialize all gates at 0.5 (sigmoid(0))
    gates[0, :] = 0.5
    
    # Simulate gradient-flow dynamics for each gate
    for i in range(num_gates):
        # Task gradient strength decreases with gate index
        # (first few gates are most useful)
        task_strength = max(0.1, 1.0 - i * 0.08)
        
        # Sparsity gradient is constant (lambda = 1e-4 in practice)
        sparsity_strength = 0.15
        
        for t in range(1, num_steps):
            # Current gate value
            g = gates[t-1, i]
            
            # Task gradient: increases g if gate is useful
            # Decreases over time as model converges
            task_grad = task_strength * np.exp(-t / 300) * (1 - g)
            
            # Sparsity gradient: always pushes toward 0
            sparsity_grad = -sparsity_strength * g
            
            # Combined gradient update
            delta_g = task_grad + sparsity_grad
            
            # Update with small learning rate
            gates[t, i] = np.clip(gates[t-1, i] + 0.01 * delta_g, 0, 1)
    
    # Left panel: Individual gate trajectories
    for i in range(num_gates):
        alpha = 0.8 if i < 5 else 0.4  # Highlight top 5 gates
        linewidth = 2.0 if i < 5 else 1.0
        ax1.plot(steps, gates[:, i], alpha=alpha, linewidth=linewidth,
                label=f'Gate {i+1}' if i < 5 else '')
    
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, 
                label='Initial value (0.5)', alpha=0.6)
    ax1.axhline(y=0.0, color='gray', linestyle=':', linewidth=1.0, alpha=0.3)
    ax1.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.0, alpha=0.3)
    
    ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Gate Value $g_i$', fontsize=12, fontweight='bold')
    ax1.set_title('Individual Gate Trajectories', fontsize=13, fontweight='bold')
    ax1.legend(loc='right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Add annotation for differentiation
    ax1.annotate('Gates differentiate:\nUseful → 1.0\nUseless → 0.0', 
                xy=(600, 0.5), xytext=(750, 0.7),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    # Right panel: Effective rank over time
    effective_rank = gates.sum(axis=1)
    
    ax2.plot(steps, effective_rank, color='blue', linewidth=2.5, label='Effective Rank')
    ax2.axhline(y=num_gates/2, color='red', linestyle='--', linewidth=1.5, 
                label=f'Initial (r/2 = {num_gates/2})', alpha=0.6)
    
    # Final converged rank
    final_rank = effective_rank[-1]
    ax2.axhline(y=final_rank, color='green', linestyle='--', linewidth=1.5,
                label=f'Final (≈{final_rank:.1f})', alpha=0.6)
    
    ax2.fill_between(steps, 0, effective_rank, alpha=0.2, color='blue')
    
    ax2.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Effective Rank $r_{eff} = \\sum g_i$', fontsize=12, fontweight='bold')
    ax2.set_title('Effective Rank Convergence', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, num_gates + 1)
    
    # Add annotation for parameter savings
    savings = (1 - final_rank / num_gates) * 100
    ax2.annotate(f'{savings:.1f}% Parameter\nSavings vs Fixed Rank', 
                xy=(800, final_rank), xytext=(600, 2),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/gate_evolution.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def create_per_layer_adaptation():
    """
    Visualization 2: Per-Layer Rank Adaptation
    
    Left panel: Heatmap showing gate values for each layer
    Right panel: Effective rank by layer depth
    
    Demonstrates how different layers learn different optimal ranks.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Per-Layer Rank Adaptation in 12-Layer Transformer', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    num_layers = 12
    max_rank = 16
    
    # Generate realistic per-layer gate patterns
    # Early layers: low rank (simple features)
    # Middle layers: medium rank (syntax/grammar)
    # Late layers: high rank (semantic reasoning)
    
    gate_matrix = np.zeros((num_layers, max_rank))
    
    for layer in range(num_layers):
        # Capacity need increases with layer depth (roughly logarithmic)
        capacity_need = 0.3 + 0.6 * (np.log(layer + 1) / np.log(num_layers + 1))
        
        # Number of active gates increases with depth
        num_active = int(capacity_need * max_rank)
        
        # Generate sorted gate values (high to low)
        for i in range(max_rank):
            if i < num_active:
                # Active gates: high values with some variance
                gate_matrix[layer, i] = np.clip(
                    np.random.beta(8, 2) * (1 - i / (num_active * 1.5)), 0, 1
                )
            else:
                # Inactive gates: very low values
                gate_matrix[layer, i] = np.random.beta(1, 10) * 0.1
        
        # Sort gates in descending order for each layer
        gate_matrix[layer, :] = np.sort(gate_matrix[layer, :])[::-1]
    
    # Left panel: Heatmap
    im = ax1.imshow(gate_matrix, cmap='YlOrRd', aspect='auto', 
                    vmin=0, vmax=1, interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Gate Value $g_i$', fontsize=11, fontweight='bold')
    
    ax1.set_xlabel('Gate Index (i)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Layer Depth', fontsize=12, fontweight='bold')
    ax1.set_title('Gate Value Heatmap Across Layers', fontsize=13, fontweight='bold')
    
    # Set ticks
    ax1.set_xticks(np.arange(0, max_rank, 2))
    ax1.set_xticklabels(np.arange(1, max_rank + 1, 2))
    ax1.set_yticks(np.arange(num_layers))
    ax1.set_yticklabels([f'Layer {i+1}' for i in range(num_layers)])
    
    # Add grid
    ax1.set_xticks(np.arange(max_rank) - 0.5, minor=True)
    ax1.set_yticks(np.arange(num_layers) - 0.5, minor=True)
    ax1.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Annotate regions
    ax1.text(-1.5, 2, 'Low-level\nfeatures\n(low rank)', 
            fontsize=9, color='blue', fontweight='bold', 
            ha='right', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))
    
    ax1.text(-1.5, 6, 'Mid-level\nfeatures\n(medium rank)', 
            fontsize=9, color='orange', fontweight='bold', 
            ha='right', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.5))
    
    ax1.text(-1.5, 10, 'High-level\nsemantics\n(high rank)', 
            fontsize=9, color='red', fontweight='bold', 
            ha='right', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.5))
    
    # Right panel: Effective rank by layer
    effective_ranks = gate_matrix.sum(axis=1)
    
    colors = ['blue'] * 4 + ['orange'] * 4 + ['red'] * 4
    bars = ax2.barh(np.arange(num_layers), effective_ranks, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, rank) in enumerate(zip(bars, effective_ranks)):
        ax2.text(rank + 0.3, i, f'{rank:.1f}', 
                va='center', fontsize=9, fontweight='bold')
    
    # Reference line for maximum rank
    ax2.axvline(x=max_rank, color='red', linestyle='--', linewidth=2, 
                label=f'Max Rank ({max_rank})', alpha=0.6)
    
    # Average effective rank
    avg_rank = effective_ranks.mean()
    ax2.axvline(x=avg_rank, color='green', linestyle='--', linewidth=2,
                label=f'Average ({avg_rank:.1f})', alpha=0.6)
    
    ax2.set_xlabel('Effective Rank $r_{eff}$', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Layer Depth', fontsize=12, fontweight='bold')
    ax2.set_title('Effective Rank by Layer', fontsize=13, fontweight='bold')
    ax2.set_yticks(np.arange(num_layers))
    ax2.set_yticklabels([f'Layer {i+1}' for i in range(num_layers)])
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, axis='x', alpha=0.3)
    ax2.set_xlim(0, max_rank + 2)
    
    # Add parameter savings annotation
    param_savings = (1 - avg_rank / max_rank) * 100
    ax2.text(max_rank * 0.5, num_layers - 0.5, 
            f'Total Parameter Savings:\n{param_savings:.1f}% vs Fixed Rank-{max_rank}',
            fontsize=11, fontweight='bold', color='green',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.6),
            ha='center')
    
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/per_layer_adaptation.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def create_sparsity_accuracy_tradeoff():
    """
    Visualization 3: Sparsity vs Accuracy Trade-off
    
    Left panel: Effect of lambda on effective rank and sparsity
    Right panel: Validation accuracy vs effective rank
    
    Shows optimal lambda selection and accuracy-efficiency Pareto frontier.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('C-LoRA Sparsity-Accuracy Trade-off', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Generate data for lambda vs effective rank
    lambdas = np.logspace(-5, -2, 50)  # 1e-5 to 1e-2
    max_rank = 16
    
    # Effective rank decreases with lambda (sparsity increases)
    effective_ranks = max_rank / (1 + 100 * lambdas)
    
    # Sparsity (fraction of gates near 0)
    sparsity = 1 - effective_ranks / max_rank
    
    # Left panel: Lambda effect
    ax1_twin = ax1.twinx()
    
    # Plot effective rank
    line1 = ax1.plot(lambdas, effective_ranks, 'b-', linewidth=2.5, 
                    label='Effective Rank', marker='o', markersize=3)
    ax1.set_xlabel('Sparsity Penalty $\\lambda$', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Effective Rank $r_{eff}$', fontsize=12, fontweight='bold', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max_rank + 1)
    
    # Plot sparsity percentage
    line2 = ax1_twin.plot(lambdas, sparsity * 100, 'r-', linewidth=2.5,
                          label='Sparsity %', marker='s', markersize=3)
    ax1_twin.set_ylabel('Sparsity (%)', fontsize=12, fontweight='bold', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    ax1_twin.set_ylim(0, 105)
    
    # Highlight optimal lambda region
    optimal_lambda_min = 5e-5
    optimal_lambda_max = 5e-4
    ax1.axvspan(optimal_lambda_min, optimal_lambda_max, alpha=0.2, color='green',
                label='Optimal Range')
    
    # Add annotation for optimal point
    optimal_idx = np.argmin(np.abs(lambdas - 1e-4))
    ax1.annotate('Recommended:\n$\\lambda = 10^{-4}$', 
                xy=(lambdas[optimal_idx], effective_ranks[optimal_idx]),
                xytext=(2e-4, 12),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, color='green', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.6))
    
    # Warning regions
    ax1.text(1e-5, 14, 'Too Low:\nNo Pruning', 
            fontsize=9, color='orange', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.4))
    
    ax1.text(3e-3, 2, 'Too High:\nOver-Pruning', 
            fontsize=9, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.4))
    
    ax1.set_title('Effect of Sparsity Penalty $\\lambda$', fontsize=13, fontweight='bold')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=10)
    
    # Right panel: Accuracy vs Effective Rank
    # Generate realistic accuracy curve (diminishing returns)
    ranks_range = np.linspace(1, max_rank, 50)
    
    # Accuracy increases logarithmically with rank (diminishing returns)
    base_accuracy = 0.75
    accuracy = base_accuracy + 0.20 * np.log(ranks_range) / np.log(max_rank)
    
    # Add some realistic variance
    accuracy += np.random.normal(0, 0.005, len(ranks_range))
    accuracy = np.clip(accuracy, 0, 1)
    
    # Pareto frontier (optimal accuracy-efficiency trade-off)
    pareto_start = 6
    pareto_end = 12
    
    ax2.plot(ranks_range, accuracy, 'b-', linewidth=2.5, label='Validation Accuracy')
    ax2.fill_between(ranks_range, accuracy - 0.01, accuracy + 0.01, alpha=0.2, color='blue')
    
    # Highlight Pareto frontier
    pareto_mask = (ranks_range >= pareto_start) & (ranks_range <= pareto_end)
    ax2.scatter(ranks_range[pareto_mask], accuracy[pareto_mask], 
               color='green', s=100, marker='*', zorder=5,
               label='Pareto Optimal Zone')
    
    # Reference lines
    ax2.axhline(y=accuracy[-1], color='red', linestyle='--', linewidth=1.5,
                label=f'Max Accuracy ({accuracy[-1]:.3f})', alpha=0.6)
    
    ax2.axvline(x=max_rank, color='gray', linestyle='--', linewidth=1.5,
                label=f'Fixed Rank ({max_rank})', alpha=0.6)
    
    ax2.set_xlabel('Effective Rank $r_{eff}$', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy vs Effective Rank', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_rank + 1)
    ax2.set_ylim(0.74, 0.97)
    
    # Annotate key points
    # Sweet spot
    sweet_spot_rank = 8
    sweet_spot_idx = np.argmin(np.abs(ranks_range - sweet_spot_rank))
    sweet_spot_acc = accuracy[sweet_spot_idx]
    
    ax2.annotate(f'Sweet Spot:\nRank ≈ {sweet_spot_rank}\nAccuracy: {sweet_spot_acc:.3f}\n50% Parameters Saved', 
                xy=(sweet_spot_rank, sweet_spot_acc),
                xytext=(3, 0.88),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.6))
    
    # Diminishing returns region
    ax2.text(13, 0.92, 'Diminishing\nReturns', 
            fontsize=9, color='orange', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.4))
    
    # Underfit region
    ax2.text(2, 0.80, 'Underfit:\nToo Sparse', 
            fontsize=9, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.4))
    
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/sparsity_accuracy_tradeoff.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    print("=" * 60)
    print("Generating C-LoRA Visualizations...")
    print("=" * 60)
    
    print("\n[1/3] Creating gate evolution visualization...")
    create_gate_evolution()
    
    print("\n[2/3] Creating per-layer adaptation visualization...")
    create_per_layer_adaptation()
    
    print("\n[3/3] Creating sparsity-accuracy trade-off visualization...")
    create_sparsity_accuracy_tradeoff()
    
    print("\n" + "=" * 60)
    print("✓ All C-LoRA visualizations generated successfully!")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print("=" * 60)
    
    print("\nGenerated files:")
    print(f"  1. {OUTPUT_DIR}/gate_evolution.png")
    print(f"  2. {OUTPUT_DIR}/per_layer_adaptation.png")
    print(f"  3. {OUTPUT_DIR}/sparsity_accuracy_tradeoff.png")
    print("\nTotal: 3 visualizations at 300 DPI")
