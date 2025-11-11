"""
Section 11: Safety, Ethics, and Bias Mitigation - Bias Detection Visualizations
Creates 2 visualizations for bias detection methods (WEAT and Counterfactual Analysis)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
OUTPUT_DIR = "../../images/safety"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# High-resolution publication quality
DPI = 300

def create_bias_detection_visualization():
    """
    Create 2-panel visualization for bias detection methods:
    1. WEAT (Word Embedding Association Test) scores across demographic groups
    2. Counterfactual bias analysis with perplexity differences
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Bias Detection Methods: WEAT and Counterfactual Analysis', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Panel 1: WEAT Scores - Gender and Race bias in embeddings
    ax1 = axes[0]
    
    # Simulate WEAT effect sizes for different attribute pairs
    # Positive = bias toward group 1, Negative = bias toward group 2
    attributes = [
        'Career vs\nFamily',
        'Science vs\nArts',
        'Leadership vs\nSupport',
        'Math vs\nLanguage',
        'Professional vs\nDomestic'
    ]
    
    # Effect sizes for gender (male vs female names)
    gender_effect_sizes = [0.82, 0.65, 0.71, 0.58, 0.76]
    
    # Effect sizes for race (European vs African American names)
    race_effect_sizes = [0.48, 0.41, 0.52, 0.38, 0.44]
    
    x = np.arange(len(attributes))
    width = 0.35
    
    bars1 = ax1.barh(x - width/2, gender_effect_sizes, width, 
                     label='Gender Bias (M vs F)', color='#FF6B6B', alpha=0.8)
    bars2 = ax1.barh(x + width/2, race_effect_sizes, width,
                     label='Race Bias (Eur vs AA)', color='#4ECDC4', alpha=0.8)
    
    # Add significance threshold line
    ax1.axvline(x=0.4, color='red', linestyle='--', linewidth=2, 
               label='Significance (d>0.4)', alpha=0.7)
    
    # Add value labels
    for i, (v1, v2) in enumerate(zip(gender_effect_sizes, race_effect_sizes)):
        ax1.text(v1 + 0.02, i - width/2, f'{v1:.2f}', 
                va='center', fontsize=9, fontweight='bold')
        ax1.text(v2 + 0.02, i + width/2, f'{v2:.2f}', 
                va='center', fontsize=9, fontweight='bold')
    
    ax1.set_yticks(x)
    ax1.set_yticklabels(attributes, fontsize=10)
    ax1.set_xlabel('WEAT Effect Size (Cohen\'s d)', fontsize=11, fontweight='bold')
    ax1.set_title('(1) Word Embedding Association Test (WEAT)\nBias in Attribute Associations', 
                  fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_xlim(0, 1.0)
    
    # Add interpretation box
    textstr = 'Interpretation:\nd > 0.4: Significant bias\nd > 0.8: Strong bias\nPositive: Favors Group 1'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax1.text(0.98, 0.98, textstr, transform=ax1.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Panel 2: Counterfactual Bias Analysis
    ax2 = axes[1]
    
    # Simulate perplexity differences for counterfactual pairs
    templates = [
        'The [X] is a\nsuccessful\nprofessional',
        '[X] excels at\ntechnical tasks',
        'The nurse said\n[X] would help',
        '[X] enjoys\ncooking at home',
        'The CEO\nannounced [X]\nwill lead'
    ]
    
    # Perplexity differences (|P(male) - P(female)|)
    perplexity_diffs = [3.2, 2.8, 1.5, 2.1, 3.5]
    
    # Color by severity
    colors = ['red' if d > 3.0 else 'orange' if d > 2.0 else 'green' 
              for d in perplexity_diffs]
    
    bars = ax2.bar(range(len(templates)), perplexity_diffs, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, perplexity_diffs)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xticks(range(len(templates)))
    ax2.set_xticklabels(templates, fontsize=9, rotation=0, ha='center')
    ax2.set_ylabel('Perplexity Difference\n|PPL(male) - PPL(female)|', 
                   fontsize=11, fontweight='bold')
    ax2.set_title('(2) Counterfactual Bias Analysis\nPerplexity Gaps Across Gender Swaps', 
                  fontsize=12, fontweight='bold', pad=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 4.5)
    
    # Add threshold lines
    ax2.axhline(y=2.0, color='orange', linestyle='--', linewidth=2, 
               label='Moderate Bias (Δ>2.0)', alpha=0.6)
    ax2.axhline(y=3.0, color='red', linestyle='--', linewidth=2,
               label='High Bias (Δ>3.0)', alpha=0.6)
    ax2.legend(loc='upper left', fontsize=9)
    
    # Add interpretation box
    textstr = 'Color Code:\nRed: High bias (Δ>3.0)\nOrange: Moderate (Δ>2.0)\nGreen: Low (Δ<2.0)'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.3)
    ax2.text(0.98, 0.50, textstr, transform=ax2.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/bias_detection_methods.png", dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/bias_detection_methods.png")
    plt.close()

if __name__ == "__main__":
    print("Generating Section 11 visualizations: Bias Detection...")
    create_bias_detection_visualization()
    print("\n✓ All bias detection visualizations generated successfully!")
    print(f"✓ Output directory: {OUTPUT_DIR}")
