"""
Advanced Visualizations for Section 9: Instruction Fine-Tuning and SFT Datasets
Creates professional, publication-quality figures with clear, readable layouts
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle, Wedge
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os

# Set style for professional appearance
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Output directory
OUTPUT_DIR = "../images/instruction_finetuning"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# High-quality settings
DPI = 300
FIGSIZE_WIDE = (14, 8)
FIGSIZE_SQUARE = (10, 10)
FIGSIZE_TALL = (12, 10)

def create_traditional_vs_instruction_comparison():
    """
    Figure 1: Traditional Fine-Tuning vs Instruction Fine-Tuning Architecture
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Traditional Fine-Tuning (Left)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Traditional Fine-Tuning\n(Single Task)', fontsize=16, fontweight='bold', pad=20)

    # Input layer
    input_box = FancyBboxPatch((1, 7.5), 8, 1, boxstyle="round,pad=0.1",
                               edgecolor='#2E86AB', facecolor='#A7C6DA', linewidth=2)
    ax1.add_patch(input_box)
    ax1.text(5, 8, 'Input: "Hello, how are you?"', ha='center', va='center',
             fontsize=11, fontweight='bold')

    # Model
    model_box = FancyBboxPatch((2, 4.5), 6, 2, boxstyle="round,pad=0.1",
                               edgecolor='#F77F00', facecolor='#FCBF49', linewidth=2)
    ax1.add_patch(model_box)
    ax1.text(5, 5.5, 'Pre-trained LLM\n(GPT-2, LLaMA)', ha='center', va='center',
             fontsize=12, fontweight='bold')

    # Output
    output_box = FancyBboxPatch((1, 1.5), 8, 1, boxstyle="round,pad=0.1",
                                edgecolor='#06A77D', facecolor='#90E39A', linewidth=2)
    ax1.add_patch(output_box)
    ax1.text(5, 2, 'Output: Translation/Summary/QA', ha='center', va='center',
             fontsize=11, fontweight='bold')

    # Arrows
    ax1.annotate('', xy=(5, 7.5), xytext=(5, 6.5),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
    ax1.annotate('', xy=(5, 4.5), xytext=(5, 2.5),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

    # Task label
    task_label = FancyBboxPatch((0.5, 0.3), 9, 0.8, boxstyle="round,pad=0.05",
                                edgecolor='#D62828', facecolor='#F77F00', linewidth=2, alpha=0.7)
    ax1.add_patch(task_label)
    ax1.text(5, 0.7, 'Optimized for ONE task only', ha='center', va='center',
             fontsize=11, fontweight='bold', color='white')

    # Instruction Fine-Tuning (Right)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Instruction Fine-Tuning\n(Multi-Task)', fontsize=16, fontweight='bold', pad=20)

    # Instruction layer
    inst_box = FancyBboxPatch((0.5, 8.5), 9, 0.8, boxstyle="round,pad=0.1",
                              edgecolor='#7209B7', facecolor='#B5A2CA', linewidth=2)
    ax2.add_patch(inst_box)
    ax2.text(5, 8.9, 'Instruction: "Translate to French"', ha='center', va='center',
             fontsize=11, fontweight='bold')

    # Input layer
    input_box2 = FancyBboxPatch((1, 7), 8, 0.8, boxstyle="round,pad=0.1",
                                edgecolor='#2E86AB', facecolor='#A7C6DA', linewidth=2)
    ax2.add_patch(input_box2)
    ax2.text(5, 7.4, 'Input: "Hello, how are you?"', ha='center', va='center',
             fontsize=11, fontweight='bold')

    # Model
    model_box2 = FancyBboxPatch((2, 4), 6, 2, boxstyle="round,pad=0.1",
                                edgecolor='#F77F00', facecolor='#FCBF49', linewidth=2)
    ax2.add_patch(model_box2)
    ax2.text(5, 5, 'Pre-trained LLM\n(Instruction-Aware)', ha='center', va='center',
             fontsize=12, fontweight='bold')

    # Output
    output_box2 = FancyBboxPatch((1, 1.5), 8, 1, boxstyle="round,pad=0.1",
                                 edgecolor='#06A77D', facecolor='#90E39A', linewidth=2)
    ax2.add_patch(output_box2)
    ax2.text(5, 2, 'Output: "Bonjour, comment allez-vous?"', ha='center', va='center',
             fontsize=11, fontweight='bold')

    # Arrows
    ax2.annotate('', xy=(5, 7), xytext=(5, 6),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
    ax2.annotate('', xy=(5, 4), xytext=(5, 2.5),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

    # Multi-task capability boxes
    tasks = ['Translation', 'Summarization', 'QA', 'Code Gen', 'Math', 'Writing']
    colors = ['#E63946', '#F77F00', '#FCBF49', '#06A77D', '#118AB2', '#7209B7']

    for i, (task, color) in enumerate(zip(tasks, colors)):
        x = 0.5 + (i % 3) * 3
        y = 0.3 if i < 3 else -0.5
        task_box = FancyBboxPatch((x, y), 2.8, 0.5, boxstyle="round,pad=0.05",
                                  edgecolor=color, facecolor=color, linewidth=1.5, alpha=0.7)
        ax2.add_patch(task_box)
        ax2.text(x + 1.4, y + 0.25, task, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/traditional_vs_instruction.png", dpi=DPI, bbox_inches='tight')
    print("✓ Created: traditional_vs_instruction.png")
    plt.close()


def create_dataset_format_comparison():
    """
    Figure 2: Popular Instruction Dataset Formats Comparison
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    formats = [
        {
            'name': 'Alpaca Format',
            'structure': '{\n  "instruction": "Task description",\n  "input": "Context/data",\n  "output": "Expected response"\n}',
            'examples': '52K',
            'color': '#E63946'
        },
        {
            'name': 'ShareGPT Format',
            'structure': '{\n  "conversations": [\n    {"from": "human", "value": "..."},\n    {"from": "gpt", "value": "..."}\n  ]\n}',
            'examples': '90K',
            'color': '#F77F00'
        },
        {
            'name': 'FLAN Format',
            'structure': '{\n  "task": "Task type",\n  "inputs": "Input text",\n  "targets": "Output text"\n}',
            'examples': '1.8M',
            'color': '#FCBF49'
        },
        {
            'name': 'Dolly Format',
            'structure': '{\n  "instruction": "Task",\n  "context": "Background",\n  "response": "Answer",\n  "category": "Type"\n}',
            'examples': '15K',
            'color': '#06A77D'
        },
        {
            'name': 'OpenOrca Format',
            'structure': '{\n  "system_prompt": "Role",\n  "question": "User query",\n  "response": "Assistant reply"\n}',
            'examples': '4.2M',
            'color': '#118AB2'
        },
        {
            'name': 'WizardLM Format',
            'structure': '{\n  "conversations": [\n    {"role": "system", "content": "..."},\n    {"role": "user", "content": "..."},\n    {"role": "assistant", "content": "..."}\n  ]\n}',
            'examples': '250K',
            'color': '#7209B7'
        }
    ]

    for idx, fmt in enumerate(formats):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        # Title box
        title_box = FancyBboxPatch((0.5, 8.5), 9, 1.2, boxstyle="round,pad=0.1",
                                   edgecolor=fmt['color'], facecolor=fmt['color'],
                                   linewidth=2.5, alpha=0.9)
        ax.add_patch(title_box)
        ax.text(5, 9.1, fmt['name'], ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')

        # Structure box
        struct_box = FancyBboxPatch((0.5, 2.5), 9, 5.5, boxstyle="round,pad=0.1",
                                    edgecolor=fmt['color'], facecolor='#F8F9FA',
                                    linewidth=2)
        ax.add_patch(struct_box)
        ax.text(5, 5.5, fmt['structure'], ha='center', va='center',
                fontsize=9, family='monospace', multialignment='left')

        # Stats box
        stats_box = FancyBboxPatch((0.5, 0.5), 9, 1.5, boxstyle="round,pad=0.1",
                                   edgecolor=fmt['color'], facecolor=fmt['color'],
                                   linewidth=2, alpha=0.3)
        ax.add_patch(stats_box)
        ax.text(5, 1.25, f'Examples: {fmt["examples"]}', ha='center', va='center',
                fontsize=11, fontweight='bold')

    fig.suptitle('Popular Instruction Dataset Formats', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(f"{OUTPUT_DIR}/dataset_formats.png", dpi=DPI, bbox_inches='tight')
    print("✓ Created: dataset_formats.png")
    plt.close()


def create_instruction_quality_metrics():
    """
    Figure 3: Instruction Quality Dimensions and Scoring
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Quality Dimensions Radar Chart
    categories = ['Clarity', 'Specificity', 'Complexity', 'Diversity', 'Coherence', 'Relevance']

    high_quality = [9, 8.5, 7, 9, 9.5, 9]
    low_quality = [5, 4, 3, 4, 5, 4]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    high_quality += high_quality[:1]
    low_quality += low_quality[:1]
    angles += angles[:1]

    ax1 = plt.subplot(221, projection='polar')
    ax1.plot(angles, high_quality, 'o-', linewidth=2.5, label='High Quality', color='#06A77D')
    ax1.fill(angles, high_quality, alpha=0.25, color='#06A77D')
    ax1.plot(angles, low_quality, 'o-', linewidth=2.5, label='Low Quality', color='#E63946')
    ax1.fill(angles, low_quality, alpha=0.25, color='#E63946')

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 10)
    ax1.set_yticks([2, 4, 6, 8, 10])
    ax1.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax1.set_title('Quality Dimensions Comparison', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 2. Dataset Size vs Performance
    ax2.set_title('Dataset Size Impact on Model Performance', fontsize=14, fontweight='bold', pad=15)

    dataset_sizes = np.array([1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000])
    performance = 100 * (1 - np.exp(-dataset_sizes / 75000))  # Logarithmic growth

    ax2.plot(dataset_sizes / 1000, performance, 'o-', linewidth=3, markersize=8,
             color='#118AB2', label='Accuracy (%)')
    ax2.fill_between(dataset_sizes / 1000, performance, alpha=0.3, color='#118AB2')

    # Annotate key points
    key_points = [(10, performance[2]), (50, performance[4]), (500, performance[7])]
    for x, y in key_points:
        ax2.annotate(f'{y:.1f}%', xy=(x, y), xytext=(x, y+5),
                    ha='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    ax2.set_xlabel('Dataset Size (thousands of examples)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Model Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 550)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=11, loc='lower right')

    # 3. Task Diversity Distribution
    ax3.set_title('Task Category Distribution in FLAN Dataset', fontsize=14, fontweight='bold', pad=15)

    tasks = ['QA', 'Translation', 'Summarization', 'Classification', 'Code Gen',
             'Math', 'Writing', 'Other']
    percentages = [25, 18, 15, 12, 10, 8, 7, 5]
    colors = ['#E63946', '#F77F00', '#FCBF49', '#06A77D', '#118AB2', '#7209B7', '#D62828', '#ADB5BD']

    wedges, texts, autotexts = ax3.pie(percentages, labels=tasks, autopct='%1.1f%%',
                                        colors=colors, startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    # 4. Instruction Length Distribution
    ax4.set_title('Instruction Length Distribution', fontsize=14, fontweight='bold', pad=15)

    # Simulate instruction length data
    np.random.seed(42)
    lengths = np.concatenate([
        np.random.normal(30, 10, 1000),   # Short instructions
        np.random.normal(80, 20, 1500),   # Medium instructions
        np.random.normal(150, 30, 500)    # Long instructions
    ])
    lengths = lengths[lengths > 0]  # Remove negatives

    ax4.hist(lengths, bins=40, color='#118AB2', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax4.axvline(np.mean(lengths), color='#E63946', linestyle='--', linewidth=2.5,
                label=f'Mean: {np.mean(lengths):.1f} tokens')
    ax4.axvline(np.median(lengths), color='#06A77D', linestyle='--', linewidth=2.5,
                label=f'Median: {np.median(lengths):.1f} tokens')

    ax4.set_xlabel('Instruction Length (tokens)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=11, loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/instruction_quality_metrics.png", dpi=DPI, bbox_inches='tight')
    print("✓ Created: instruction_quality_metrics.png")
    plt.close()


def create_data_augmentation_pipeline():
    """
    Figure 4: Instruction Data Augmentation Pipeline
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title('Instruction Data Augmentation Pipeline', fontsize=18, fontweight='bold', pad=20)

    # Original instruction
    orig_box = FancyBboxPatch((1, 9.5), 3.5, 1.2, boxstyle="round,pad=0.1",
                              edgecolor='#2E86AB', facecolor='#A7C6DA', linewidth=2.5)
    ax.add_patch(orig_box)
    ax.text(2.75, 10.1, 'Original\nInstruction', ha='center', va='center',
            fontsize=11, fontweight='bold')

    # Augmentation techniques
    techniques = [
        {'name': 'Paraphrasing', 'y': 7, 'color': '#E63946', 'desc': 'Reword using\nsynonyms'},
        {'name': 'Complexity\nVariation', 'y': 4.5, 'color': '#F77F00', 'desc': 'Simplify or\nelaborate'},
        {'name': 'Format\nConversion', 'y': 2, 'color': '#FCBF49', 'desc': 'Change style\nor structure'}
    ]

    for i, tech in enumerate(techniques):
        # Technique box
        tech_box = FancyBboxPatch((5.5, tech['y']), 3.5, 1.2, boxstyle="round,pad=0.1",
                                  edgecolor=tech['color'], facecolor=tech['color'],
                                  linewidth=2.5, alpha=0.8)
        ax.add_patch(tech_box)
        ax.text(7.25, tech['y'] + 0.6, tech['name'], ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

        # Arrow from original
        ax.annotate('', xy=(5.5, tech['y'] + 0.6), xytext=(4.5, 10.1),
                   arrowprops=dict(arrowstyle='->', lw=2, color=tech['color']))

        # Description
        desc_box = FancyBboxPatch((9.5, tech['y'] + 0.15), 2.5, 0.9, boxstyle="round,pad=0.05",
                                  edgecolor=tech['color'], facecolor='white', linewidth=1.5)
        ax.add_patch(desc_box)
        ax.text(10.75, tech['y'] + 0.6, tech['desc'], ha='center', va='center',
                fontsize=9, style='italic')

        # Augmented output
        aug_box = FancyBboxPatch((12.5, tech['y']), 3, 1.2, boxstyle="round,pad=0.1",
                                 edgecolor='#06A77D', facecolor='#90E39A', linewidth=2)
        ax.add_patch(aug_box)
        ax.text(14, tech['y'] + 0.6, f'Augmented\nVersion {i+1}', ha='center', va='center',
                fontsize=10, fontweight='bold')

        # Arrow to output
        ax.annotate('', xy=(12.5, tech['y'] + 0.6), xytext=(12, tech['y'] + 0.6),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#06A77D'))

    # Self-Instruct box
    self_instruct = FancyBboxPatch((1, 0.5), 4, 1, boxstyle="round,pad=0.1",
                                   edgecolor='#7209B7', facecolor='#B5A2CA', linewidth=2.5)
    ax.add_patch(self_instruct)
    ax.text(3, 1, 'Self-Instruct:\nLLM generates\nnew instructions', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # Back-Translation box
    back_trans = FancyBboxPatch((6, 0.5), 4, 1, boxstyle="round,pad=0.1",
                                edgecolor='#118AB2', facecolor='#A7C6DA', linewidth=2.5)
    ax.add_patch(back_trans)
    ax.text(8, 1, 'Back-Translation:\nTranslate to X,\nthen back', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # Quality Filter box
    quality_filter = FancyBboxPatch((11, 0.5), 4, 1, boxstyle="round,pad=0.1",
                                    edgecolor='#D62828', facecolor='#F77F00', linewidth=2.5)
    ax.add_patch(quality_filter)
    ax.text(13, 1, 'Quality Filter:\nRemove low-quality\naugmentations', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/data_augmentation_pipeline.png", dpi=DPI, bbox_inches='tight')
    print("✓ Created: data_augmentation_pipeline.png")
    plt.close()


def create_training_dynamics():
    """
    Figure 5: Instruction Fine-Tuning Training Dynamics
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Loss curves for different dataset sizes
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Training Loss: Impact of Dataset Size', fontsize=14, fontweight='bold')

    steps = np.linspace(0, 3000, 100)

    sizes = [(1000, '#E63946'), (5000, '#F77F00'), (25000, '#FCBF49'), (100000, '#06A77D')]

    for size, color in sizes:
        # Faster convergence with more data
        decay_rate = 0.002 + (size / 1000000)
        loss = 3.5 * np.exp(-decay_rate * steps) + 0.5 + np.random.normal(0, 0.05, len(steps))
        ax1.plot(steps, loss, linewidth=2.5, label=f'{size:,} examples', color=color)

    ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 4)

    # 2. Multi-task performance over time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Multi-Task Performance Evolution', fontsize=14, fontweight='bold')

    epochs = np.arange(0, 11)

    tasks_perf = {
        'Translation': ([45, 55, 65, 72, 78, 82, 85, 87, 88, 89, 89.5], '#E63946'),
        'QA': ([40, 52, 63, 70, 76, 80, 83, 85, 86, 87, 87.5], '#F77F00'),
        'Summarization': ([38, 48, 60, 68, 74, 78, 82, 84, 85, 86, 86.5], '#FCBF49'),
        'Code Gen': ([30, 42, 55, 64, 71, 76, 80, 82, 84, 85, 85.5], '#06A77D'),
        'Math': ([25, 38, 50, 60, 68, 74, 78, 81, 83, 84, 84.5], '#118AB2')
    }

    for task, (scores, color) in tasks_perf.items():
        ax2.plot(epochs, scores, 'o-', linewidth=2.5, markersize=6, label=task, color=color)

    ax2.set_xlabel('Training Epochs', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Task Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(20, 95)

    # 3. Learning rate schedule
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title('Learning Rate Schedule with Warmup', fontsize=14, fontweight='bold')

    total_steps = 5000
    warmup_steps = 500

    steps_lr = np.arange(0, total_steps)
    max_lr = 2e-5
    min_lr = 1e-6

    # Warmup phase
    lr_schedule = np.zeros_like(steps_lr, dtype=float)
    lr_schedule[:warmup_steps] = np.linspace(0, max_lr, warmup_steps)

    # Cosine decay
    decay_steps = steps_lr[warmup_steps:]
    lr_schedule[warmup_steps:] = min_lr + 0.5 * (max_lr - min_lr) * \
                                   (1 + np.cos(np.pi * decay_steps / len(decay_steps)))

    ax3.plot(steps_lr, lr_schedule * 1e5, linewidth=2.5, color='#7209B7')
    ax3.axvline(warmup_steps, color='#E63946', linestyle='--', linewidth=2,
                label='End of Warmup')
    ax3.fill_between(steps_lr[:warmup_steps], 0, lr_schedule[:warmup_steps] * 1e5,
                     alpha=0.3, color='#FCBF49', label='Warmup Phase')

    ax3.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Learning Rate (×10⁻⁵)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(True, alpha=0.3, linestyle='--')

    # 4. Batch size vs convergence speed
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title('Batch Size Impact on Convergence', fontsize=14, fontweight='bold')

    batch_sizes = [4, 8, 16, 32, 64, 128]
    epochs_to_converge = [18, 12, 8, 6, 5, 5.5]  # Optimal at 64
    final_accuracy = [85, 87, 89, 90, 90, 89]  # Slight drop at 128

    ax4_twin = ax4.twinx()

    bar_width = 0.35
    x_pos = np.arange(len(batch_sizes))

    bars1 = ax4.bar(x_pos - bar_width/2, epochs_to_converge, bar_width,
                    label='Epochs to Converge', color='#118AB2', alpha=0.8, edgecolor='black')
    bars2 = ax4_twin.bar(x_pos + bar_width/2, final_accuracy, bar_width,
                         label='Final Accuracy', color='#06A77D', alpha=0.8, edgecolor='black')

    ax4.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Epochs to Converge', fontsize=12, fontweight='bold', color='#118AB2')
    ax4_twin.set_ylabel('Final Accuracy (%)', fontsize=12, fontweight='bold', color='#06A77D')

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(batch_sizes)
    ax4.tick_params(axis='y', labelcolor='#118AB2')
    ax4_twin.tick_params(axis='y', labelcolor='#06A77D')

    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax4.set_ylim(0, 20)
    ax4_twin.set_ylim(80, 95)

    # Combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    fig.suptitle('Instruction Fine-Tuning: Training Dynamics Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(f"{OUTPUT_DIR}/training_dynamics.png", dpi=DPI, bbox_inches='tight')
    print("✓ Created: training_dynamics.png")
    plt.close()


def create_evaluation_metrics_dashboard():
    """
    Figure 6: Comprehensive Evaluation Metrics Dashboard
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

    # 1. Task-specific accuracy comparison
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_title('Task-Specific Performance: Base vs Instruction-Tuned',
                  fontsize=14, fontweight='bold')

    tasks = ['MMLU', 'HellaSwag', 'TruthfulQA', 'GSM8K', 'HumanEval', 'BBH']
    base_scores = [45, 52, 38, 12, 8, 35]
    instruct_scores = [62, 71, 58, 45, 28, 58]

    x = np.arange(len(tasks))
    width = 0.35

    bars1 = ax1.bar(x - width/2, base_scores, width, label='Base Model',
                    color='#ADB5BD', edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, instruct_scores, width, label='Instruction-Tuned',
                    color='#06A77D', edgecolor='black', linewidth=1.5)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, fontsize=11, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim(0, 80)

    # 2. Win rate comparison
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_title('Human Preference\nWin Rate', fontsize=14, fontweight='bold')

    models = ['GPT-3.5', 'Instruction-\nTuned\nLLaMA-2']
    win_rates = [35, 65]
    colors = ['#E63946', '#06A77D']

    bars = ax2.barh(models, win_rates, color=colors, edgecolor='black', linewidth=2)

    for i, (bar, rate) in enumerate(zip(bars, win_rates)):
        ax2.text(rate + 2, i, f'{rate}%', va='center', fontsize=12, fontweight='bold')

    ax2.set_xlabel('Win Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')

    # 3-5. Capability scores (radar charts)
    capabilities_data = [
        {
            'title': 'Knowledge\nCapabilities',
            'categories': ['Factual\nAccuracy', 'Reasoning', 'Common\nSense', 'World\nKnowledge'],
            'base': [6, 5, 6, 7],
            'instruct': [8, 8, 9, 9]
        },
        {
            'title': 'Interaction\nCapabilities',
            'categories': ['Following\nInstructions', 'Context\nAwareness', 'Coherence', 'Relevance'],
            'base': [4, 5, 6, 5],
            'instruct': [9, 9, 9, 9]
        },
        {
            'title': 'Safety &\nAlignment',
            'categories': ['Truthfulness', 'Harmlessness', 'Helpfulness', 'Bias\nMitigation'],
            'base': [6, 5, 5, 4],
            'instruct': [8, 9, 9, 8]
        }
    ]

    for idx, data in enumerate(capabilities_data):
        ax = fig.add_subplot(gs[1, idx], projection='polar')

        categories = data['categories']
        base = data['base']
        instruct = data['instruct']

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        base += base[:1]
        instruct += instruct[:1]
        angles += angles[:1]

        ax.plot(angles, base, 'o-', linewidth=2, label='Base', color='#ADB5BD')
        ax.fill(angles, base, alpha=0.25, color='#ADB5BD')
        ax.plot(angles, instruct, 'o-', linewidth=2, label='Instruct', color='#06A77D')
        ax.fill(angles, instruct, alpha=0.25, color='#06A77D')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9, fontweight='bold')
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=8)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=9)
        ax.set_title(data['title'], fontsize=12, fontweight='bold', pad=15)
        ax.grid(True, linestyle='--', alpha=0.7)

    # 6-8. Distribution plots
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.set_title('Response Length\nDistribution', fontsize=12, fontweight='bold')

    np.random.seed(42)
    base_lengths = np.random.normal(50, 15, 1000)
    instruct_lengths = np.random.normal(120, 25, 1000)

    ax6.hist(base_lengths, bins=30, alpha=0.6, color='#ADB5BD', label='Base', edgecolor='black')
    ax6.hist(instruct_lengths, bins=30, alpha=0.6, color='#06A77D', label='Instruct', edgecolor='black')
    ax6.set_xlabel('Length (tokens)', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')

    ax7 = fig.add_subplot(gs[2, 1])
    ax7.set_title('Latency\nDistribution', fontsize=12, fontweight='bold')

    base_latency = np.random.gamma(4, 30, 1000)
    instruct_latency = np.random.gamma(5, 35, 1000)

    ax7.hist(base_latency, bins=30, alpha=0.6, color='#ADB5BD', label='Base', edgecolor='black')
    ax7.hist(instruct_latency, bins=30, alpha=0.6, color='#06A77D', label='Instruct', edgecolor='black')
    ax7.set_xlabel('Latency (ms)', fontsize=10, fontweight='bold')
    ax7.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3, axis='y')

    ax8 = fig.add_subplot(gs[2, 2])
    ax8.set_title('Quality Score\nDistribution', fontsize=12, fontweight='bold')

    base_quality = np.random.beta(5, 3, 1000) * 100
    instruct_quality = np.random.beta(8, 2, 1000) * 100

    ax8.hist(base_quality, bins=30, alpha=0.6, color='#ADB5BD', label='Base', edgecolor='black')
    ax8.hist(instruct_quality, bins=30, alpha=0.6, color='#06A77D', label='Instruct', edgecolor='black')
    ax8.set_xlabel('Quality Score', fontsize=10, fontweight='bold')
    ax8.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Comprehensive Evaluation Metrics Dashboard',
                 fontsize=18, fontweight='bold', y=0.995)

    plt.savefig(f"{OUTPUT_DIR}/evaluation_metrics_dashboard.png", dpi=DPI, bbox_inches='tight')
    print("✓ Created: evaluation_metrics_dashboard.png")
    plt.close()


# Generate all visualizations
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Generating Section 9 Visualizations: Instruction Fine-Tuning")
    print("="*60 + "\n")

    create_traditional_vs_instruction_comparison()
    create_dataset_format_comparison()
    create_instruction_quality_metrics()
    create_data_augmentation_pipeline()
    create_training_dynamics()
    create_evaluation_metrics_dashboard()

    print("\n" + "="*60)
    print("✓ All Section 9 visualizations generated successfully!")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print("="*60 + "\n")
