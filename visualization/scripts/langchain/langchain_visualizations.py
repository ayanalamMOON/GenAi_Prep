"""
LangChain Visualization Script
Generates publication-quality visualizations for LangChain section of LLM Study Material
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

# Output directory
output_dir = Path(__file__).parent.parent / 'images' / 'langchain'
output_dir.mkdir(parents=True, exist_ok=True)

def create_langchain_architecture():
    """Core LangChain components architecture"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.5, 'LangChain Architecture: Core Components',
            fontsize=14, weight='bold', ha='center')

    # Models layer (bottom)
    models_box = FancyBboxPatch((0.5, 0.5), 13, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#e8f4f8', edgecolor='#2E86AB', linewidth=2)
    ax.add_patch(models_box)
    ax.text(7, 1.7, 'Models Layer', fontsize=11, weight='bold', ha='center')
    ax.text(3, 1.2, 'OpenAI GPT', fontsize=9, ha='center')
    ax.text(5.5, 1.2, 'Anthropic Claude', fontsize=9, ha='center')
    ax.text(8, 1.2, 'Google PaLM', fontsize=9, ha='center')
    ax.text(10.5, 1.2, 'Local Models', fontsize=9, ha='center')
    ax.text(7, 0.7, 'Unified Interface: LLM / ChatModel', fontsize=8, ha='center', style='italic')

    # Prompts layer
    prompts_box = FancyBboxPatch((0.5, 2.3), 4, 1.2, boxstyle="round,pad=0.05",
                                 facecolor='#fff3cd', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(prompts_box)
    ax.text(2.5, 3.2, 'Prompts', fontsize=10, weight='bold', ha='center')
    ax.text(2.5, 2.85, 'Templates', fontsize=8, ha='center')
    ax.text(2.5, 2.55, 'Few-Shot', fontsize=8, ha='center')

    # Chains layer
    chains_box = FancyBboxPatch((5, 2.3), 4, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#d4edda', edgecolor='#28a745', linewidth=2)
    ax.add_patch(chains_box)
    ax.text(7, 3.2, 'Chains', fontsize=10, weight='bold', ha='center')
    ax.text(7, 2.85, 'Sequential', fontsize=8, ha='center')
    ax.text(7, 2.55, 'Router', fontsize=8, ha='center')

    # Memory layer
    memory_box = FancyBboxPatch((9.5, 2.3), 4, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#f8d7da', edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(memory_box)
    ax.text(11.5, 3.2, 'Memory', fontsize=10, weight='bold', ha='center')
    ax.text(11.5, 2.85, 'Buffer', fontsize=8, ha='center')
    ax.text(11.5, 2.55, 'Vector Store', fontsize=8, ha='center')

    # Agents layer
    agents_box = FancyBboxPatch((2, 4), 5, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=2)
    ax.add_patch(agents_box)
    ax.text(4.5, 5.2, 'Agents', fontsize=11, weight='bold', ha='center')
    ax.text(4.5, 4.7, 'ReAct', fontsize=9, ha='center')
    ax.text(4.5, 4.3, 'Plan-Execute', fontsize=9, ha='center')

    # Tools layer
    tools_box = FancyBboxPatch((7.5, 4), 5.5, 1.5, boxstyle="round,pad=0.1",
                               facecolor='#f3e5f5', edgecolor='#9c27b0', linewidth=2)
    ax.add_patch(tools_box)
    ax.text(10.25, 5.2, 'Tools & Integrations', fontsize=11, weight='bold', ha='center')
    ax.text(8.5, 4.7, 'Search', fontsize=8, ha='center')
    ax.text(9.8, 4.7, 'Calculator', fontsize=8, ha='center')
    ax.text(11.1, 4.7, 'APIs', fontsize=8, ha='center')
    ax.text(12.4, 4.7, 'Databases', fontsize=8, ha='center')

    # Application layer (top)
    app_box = FancyBboxPatch((1, 6), 12, 1, boxstyle="round,pad=0.1",
                             facecolor='#fff9c4', edgecolor='#fbc02d', linewidth=3)
    ax.add_patch(app_box)
    ax.text(7, 6.5, 'LangChain Application', fontsize=12, weight='bold', ha='center')

    # Arrows connecting layers
    # Models to components
    for x in [2.5, 7, 11.5]:
        ax.annotate('', xy=(x, 2.2), xytext=(x, 2.0),
                   arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

    # Components to Agents/Tools
    ax.annotate('', xy=(4.5, 3.9), xytext=(2.5, 3.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.annotate('', xy=(4.5, 3.9), xytext=(7, 3.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.annotate('', xy=(10.25, 3.9), xytext=(11.5, 3.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

    # To Application
    ax.annotate('', xy=(7, 5.9), xytext=(4.5, 5.5),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
    ax.annotate('', xy=(7, 5.9), xytext=(10.25, 5.5),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

    plt.tight_layout()
    plt.savefig(output_dir / 'langchain_architecture.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: langchain_architecture.png")

def create_chain_types():
    """Different types of chains visualization"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(7, 9.5, 'LangChain: Chain Types and Patterns',
            fontsize=14, weight='bold', ha='center')

    # 1. Sequential Chain
    ax.text(2, 8.5, '1. Sequential Chain', fontsize=11, weight='bold')

    seq_steps = ['Input', 'LLM Call 1', 'Process', 'LLM Call 2', 'Output']
    for i, step in enumerate(seq_steps):
        x = 0.5 + i * 0.7
        y = 7.5
        color = '#3498db' if 'LLM' in step else '#95a5a6'
        box = FancyBboxPatch((x, y), 0.6, 0.5, boxstyle="round,pad=0.03",
                            facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.7)
        ax.add_patch(box)
        ax.text(x + 0.3, y + 0.25, step, fontsize=7, ha='center', va='center',
               color='white', weight='bold')

        if i < len(seq_steps) - 1:
            ax.annotate('', xy=(x + 0.65, y + 0.25), xytext=(x + 0.6, y + 0.25),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # 2. Router Chain
    ax.text(6, 8.5, '2. Router Chain', fontsize=11, weight='bold')

    input_box = FancyBboxPatch((6.5, 7.5), 0.7, 0.5, boxstyle="round,pad=0.03",
                               facecolor='#95a5a6', edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.add_patch(input_box)
    ax.text(6.85, 7.75, 'Input', fontsize=7, ha='center', color='white', weight='bold')

    router = FancyBboxPatch((6.3, 6.5), 1.1, 0.6, boxstyle="round,pad=0.05",
                           facecolor='#f39c12', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(router)
    ax.text(6.85, 6.8, 'Router', fontsize=8, ha='center', weight='bold')

    # Routes
    routes = ['Math', 'Code', 'Text']
    colors_r = ['#e74c3c', '#27ae60', '#9b59b6']
    for i, (route, color) in enumerate(zip(routes, colors_r)):
        x = 5.5 + i * 1.2
        y = 5.5
        box = FancyBboxPatch((x, y), 0.8, 0.4, boxstyle="round,pad=0.03",
                            facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.7)
        ax.add_patch(box)
        ax.text(x + 0.4, y + 0.2, route, fontsize=7, ha='center', color='white', weight='bold')

        # Arrow from router
        ax.annotate('', xy=(x + 0.4, y + 0.42), xytext=(6.85, 6.48),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

    # 3. Map-Reduce Chain
    ax.text(10, 8.5, '3. Map-Reduce Chain', fontsize=11, weight='bold')

    input_mr = FancyBboxPatch((11, 7.5), 0.8, 0.4, boxstyle="round,pad=0.03",
                             facecolor='#95a5a6', edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.add_patch(input_mr)
    ax.text(11.4, 7.7, 'Documents', fontsize=7, ha='center', color='white', weight='bold')

    # Map phase
    for i in range(3):
        x = 10.3 + i * 0.9
        y = 6.5
        box = FancyBboxPatch((x, y), 0.7, 0.4, boxstyle="round,pad=0.03",
                            facecolor='#3498db', edgecolor='black', linewidth=1.5, alpha=0.7)
        ax.add_patch(box)
        ax.text(x + 0.35, y + 0.2, f'Map {i+1}', fontsize=6, ha='center',
               color='white', weight='bold')

        # Arrow from input
        ax.annotate('', xy=(x + 0.35, y + 0.42), xytext=(11.4, 7.48),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

    # Reduce phase
    reduce_box = FancyBboxPatch((10.8, 5.5), 1.2, 0.5, boxstyle="round,pad=0.05",
                               facecolor='#27ae60', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(reduce_box)
    ax.text(11.4, 5.75, 'Reduce', fontsize=8, ha='center', color='white', weight='bold')

    # Arrows to reduce
    for i in range(3):
        x = 10.3 + i * 0.9 + 0.35
        ax.annotate('', xy=(11.4, 6.0), xytext=(x, 6.48),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

    # 4. Conversational Chain
    ax.text(1, 4.8, '4. Conversational Chain with Memory', fontsize=11, weight='bold')

    # User-Assistant exchange
    for i in range(3):
        y_base = 3.8 - i * 0.9

        # User message
        user_box = FancyBboxPatch((0.5, y_base), 1.5, 0.35, boxstyle="round,pad=0.03",
                                 facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=1.5)
        ax.add_patch(user_box)
        ax.text(1.25, y_base + 0.175, f'User {i+1}', fontsize=7, ha='center', weight='bold')

        # Assistant response
        assist_box = FancyBboxPatch((2.3, y_base), 1.5, 0.35, boxstyle="round,pad=0.03",
                                   facecolor='#f3e5f5', edgecolor='#9c27b0', linewidth=1.5)
        ax.add_patch(assist_box)
        ax.text(3.05, y_base + 0.175, f'Assistant {i+1}', fontsize=7, ha='center', weight='bold')

        # Arrow
        ax.annotate('', xy=(2.28, y_base + 0.175), xytext=(2.02, y_base + 0.175),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Memory component
    memory_box = FancyBboxPatch((4.2, 2.5), 1.2, 1.2, boxstyle="round,pad=0.05",
                               facecolor='#fff3cd', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(memory_box)
    ax.text(4.8, 3.5, 'Memory', fontsize=9, ha='center', weight='bold')
    ax.text(4.8, 3.15, 'Buffer', fontsize=7, ha='center')
    ax.text(4.8, 2.85, 'Context', fontsize=7, ha='center')

    # Arrows to memory
    for i in range(3):
        y = 3.8 - i * 0.9 + 0.175
        ax.annotate('', xy=(4.18, 3.1), xytext=(3.85, y),
                   arrowprops=dict(arrowstyle='->', lw=1, color='gray', alpha=0.6))

    # 5. RAG Chain
    ax.text(7, 4.8, '5. Retrieval-Augmented Generation (RAG)', fontsize=11, weight='bold')

    query_box = FancyBboxPatch((7, 3.8), 0.8, 0.35, boxstyle="round,pad=0.03",
                              facecolor='#95a5a6', edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.add_patch(query_box)
    ax.text(7.4, 3.975, 'Query', fontsize=7, ha='center', color='white', weight='bold')

    # Vector DB
    vdb_box = FancyBboxPatch((6.8, 2.8), 1.2, 0.5, boxstyle="round,pad=0.05",
                            facecolor='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.add_patch(vdb_box)
    ax.text(7.4, 3.05, 'Vector DB', fontsize=7, ha='center', color='white', weight='bold')

    # Retrieved docs
    docs_box = FancyBboxPatch((8.3, 2.8), 1.2, 0.5, boxstyle="round,pad=0.03",
                             facecolor='#3498db', edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.add_patch(docs_box)
    ax.text(8.9, 3.05, 'Docs', fontsize=7, ha='center', color='white', weight='bold')

    # LLM with context
    llm_box = FancyBboxPatch((7.5, 1.8), 1.8, 0.6, boxstyle="round,pad=0.05",
                            facecolor='#27ae60', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(llm_box)
    ax.text(8.4, 2.1, 'LLM + Context', fontsize=8, ha='center', color='white', weight='bold')

    # Arrows
    ax.annotate('', xy=(7.4, 2.78), xytext=(7.4, 3.78),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(8.9, 2.78), xytext=(7.4, 2.78),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    ax.annotate('', xy=(8.4, 1.78), xytext=(7.4, 2.78),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(8.4, 1.78), xytext=(8.9, 2.78),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Output
    output_box = FancyBboxPatch((7.8, 0.8), 1.2, 0.4, boxstyle="round,pad=0.03",
                               facecolor='#9b59b6', edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.add_patch(output_box)
    ax.text(8.4, 1.0, 'Answer', fontsize=7, ha='center', color='white', weight='bold')

    ax.annotate('', xy=(8.4, 1.22), xytext=(8.4, 1.78),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#3498db', label='LLM Call'),
        mpatches.Patch(facecolor='#95a5a6', label='Input/Data'),
        mpatches.Patch(facecolor='#27ae60', label='Processing'),
        mpatches.Patch(facecolor='#f39c12', label='Memory/Router')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'chain_types.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: chain_types.png")

def create_memory_types():
    """Memory systems comparison"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Memory types comparison
    memory_types = ['Buffer\nMemory', 'Window\nMemory', 'Summary\nMemory', 'Vector\nMemory', 'Entity\nMemory']
    retention = [100, 80, 60, 95, 85]  # Percentage
    speed = [90, 95, 70, 60, 65]  # Relative speed

    x = np.arange(len(memory_types))
    width = 0.35

    bars1 = ax1.bar(x - width/2, retention, width, label='Information Retention',
                    color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, speed, width, label='Access Speed',
                    color='#e74c3c', alpha=0.8)

    ax1.set_ylabel('Score (%)', fontsize=11, weight='bold')
    ax1.set_title('Memory Types Performance Comparison', fontsize=12, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(memory_types, fontsize=9)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 110)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{int(height)}%', ha='center', va='bottom', fontsize=7)

    # 2. Memory size vs performance
    conv_length = np.array([10, 50, 100, 200, 500, 1000])
    buffer_perf = 100 - (conv_length / 10)  # Degrades quickly
    buffer_perf = np.clip(buffer_perf, 10, 100)

    window_perf = np.ones_like(conv_length) * 85  # Constant

    summary_perf = 90 - (conv_length / 50)  # Slow degradation
    summary_perf = np.clip(summary_perf, 60, 90)

    vector_perf = 95 - (conv_length / 200)  # Very slow degradation
    vector_perf = np.clip(vector_perf, 80, 95)

    ax2.plot(conv_length, buffer_perf, 'o-', linewidth=2.5, markersize=7,
            label='Buffer Memory', color='#3498db')
    ax2.plot(conv_length, window_perf, 's-', linewidth=2.5, markersize=7,
            label='Window Memory', color='#27ae60')
    ax2.plot(conv_length, summary_perf, '^-', linewidth=2.5, markersize=7,
            label='Summary Memory', color='#f39c12')
    ax2.plot(conv_length, vector_perf, 'd-', linewidth=2.5, markersize=7,
            label='Vector Memory', color='#e74c3c')

    ax2.set_xlabel('Conversation Length (messages)', fontsize=11, weight='bold')
    ax2.set_ylabel('Performance (%)', fontsize=11, weight='bold')
    ax2.set_title('Memory Performance vs Conversation Length', fontsize=12, weight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 105)

    # 3. Memory usage
    memory_sizes = ['Buffer', 'Window\n(k=10)', 'Summary', 'Vector\nStore', 'Entity']
    mem_usage_kb = [500, 150, 80, 200, 120]  # KB per 100 messages

    bars = ax3.bar(memory_sizes, mem_usage_kb, color=['#3498db', '#27ae60', '#f39c12', '#e74c3c', '#9b59b6'],
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    ax3.set_ylabel('Memory Usage (KB/100 msgs)', fontsize=11, weight='bold')
    ax3.set_title('Memory Footprint Comparison', fontsize=12, weight='bold')
    ax3.grid(axis='y', alpha=0.3)

    for bar, usage in zip(bars, mem_usage_kb):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{usage} KB', ha='center', va='bottom', fontsize=9, weight='bold')

    # 4. Use case matrix
    use_cases = ['Short\nChat', 'Long\nConv', 'QA\nSystem', 'Multi-\nTopic', 'High\nAccuracy']
    memory_scores = {
        'Buffer': [95, 40, 60, 50, 90],
        'Window': [85, 80, 70, 60, 75],
        'Summary': [70, 85, 75, 80, 65],
        'Vector': [80, 90, 95, 85, 90],
        'Entity': [75, 85, 80, 90, 85]
    }

    # Create heatmap data
    data = np.array([memory_scores[m] for m in ['Buffer', 'Window', 'Summary', 'Vector', 'Entity']])

    im = ax4.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax4.set_xticks(np.arange(len(use_cases)))
    ax4.set_yticks(np.arange(len(memory_sizes)))
    ax4.set_xticklabels(use_cases, fontsize=9)
    ax4.set_yticklabels(memory_sizes, fontsize=9)

    # Add values in cells
    for i in range(len(memory_sizes)):
        for j in range(len(use_cases)):
            text = ax4.text(j, i, f'{data[i, j]:.0f}',
                          ha="center", va="center", color="black", fontsize=8, weight='bold')

    ax4.set_title('Memory Type Suitability Matrix', fontsize=12, weight='bold')
    plt.colorbar(im, ax=ax4, label='Suitability Score')

    plt.tight_layout()
    plt.savefig(output_dir / 'memory_types.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: memory_types.png")

def create_agent_workflow():
    """Agent decision-making workflow (ReAct pattern)"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    ax.text(7, 8.5, 'LangChain Agent: ReAct (Reasoning + Acting) Pattern',
            fontsize=14, weight='bold', ha='center')

    # Initial question
    question_box = FancyBboxPatch((5.5, 7), 3, 0.6, boxstyle="round,pad=0.05",
                                  facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=2)
    ax.add_patch(question_box)
    ax.text(7, 7.3, 'User Question', fontsize=10, ha='center', weight='bold')

    # Iteration 1
    ax.text(1, 6.2, 'Iteration 1', fontsize=10, weight='bold', color='#e74c3c')

    thought1 = FancyBboxPatch((0.5, 5.2), 2, 0.7, boxstyle="round,pad=0.05",
                             facecolor='#fff3cd', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(thought1)
    ax.text(1.5, 5.7, 'Thought', fontsize=9, ha='center', weight='bold')
    ax.text(1.5, 5.4, 'Need to search', fontsize=7, ha='center', style='italic')

    action1 = FancyBboxPatch((3, 5.2), 2, 0.7, boxstyle="round,pad=0.05",
                            facecolor='#d4edda', edgecolor='#28a745', linewidth=2)
    ax.add_patch(action1)
    ax.text(4, 5.7, 'Action', fontsize=9, ha='center', weight='bold')
    ax.text(4, 5.4, 'Search("topic")', fontsize=7, ha='center', family='monospace')

    obs1 = FancyBboxPatch((5.5, 5.2), 2, 0.7, boxstyle="round,pad=0.05",
                         facecolor='#f8d7da', edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(obs1)
    ax.text(6.5, 5.7, 'Observation', fontsize=9, ha='center', weight='bold')
    ax.text(6.5, 5.4, 'Found info...', fontsize=7, ha='center', style='italic')

    # Arrows
    ax.annotate('', xy=(2.98, 5.55), xytext=(2.52, 5.55),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(5.48, 5.55), xytext=(5.02, 5.55),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Iteration 2
    ax.text(1, 4, 'Iteration 2', fontsize=10, weight='bold', color='#27ae60')

    thought2 = FancyBboxPatch((0.5, 3), 2, 0.7, boxstyle="round,pad=0.05",
                             facecolor='#fff3cd', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(thought2)
    ax.text(1.5, 3.5, 'Thought', fontsize=9, ha='center', weight='bold')
    ax.text(1.5, 3.2, 'Need calculation', fontsize=7, ha='center', style='italic')

    action2 = FancyBboxPatch((3, 3), 2, 0.7, boxstyle="round,pad=0.05",
                            facecolor='#d4edda', edgecolor='#28a745', linewidth=2)
    ax.add_patch(action2)
    ax.text(4, 3.5, 'Action', fontsize=9, ha='center', weight='bold')
    ax.text(4, 3.2, 'Calculator(expr)', fontsize=7, ha='center', family='monospace')

    obs2 = FancyBboxPatch((5.5, 3), 2, 0.7, boxstyle="round,pad=0.05",
                         facecolor='#f8d7da', edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(obs2)
    ax.text(6.5, 3.5, 'Observation', fontsize=9, ha='center', weight='bold')
    ax.text(6.5, 3.2, 'Result: 42', fontsize=7, ha='center', family='monospace')

    # Arrows
    ax.annotate('', xy=(2.98, 3.35), xytext=(2.52, 3.35),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(5.48, 3.35), xytext=(5.02, 3.35),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Vertical arrows between iterations
    ax.annotate('', xy=(1.5, 2.98), xytext=(1.5, 5.18),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray', linestyle='--'))

    # Final thought
    ax.text(1, 1.8, 'Final', fontsize=10, weight='bold', color='#9b59b6')

    final_thought = FancyBboxPatch((0.5, 0.8), 2, 0.7, boxstyle="round,pad=0.05",
                                   facecolor='#fff3cd', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(final_thought)
    ax.text(1.5, 1.3, 'Thought', fontsize=9, ha='center', weight='bold')
    ax.text(1.5, 1.0, 'I have answer', fontsize=7, ha='center', style='italic')

    final_answer = FancyBboxPatch((3, 0.8), 4, 0.7, boxstyle="round,pad=0.05",
                                  facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=2)
    ax.add_patch(final_answer)
    ax.text(5, 1.3, 'Final Answer', fontsize=9, ha='center', weight='bold')
    ax.text(5, 1.0, 'Complete response to user', fontsize=7, ha='center', style='italic')

    # Arrow
    ax.annotate('', xy=(2.98, 1.15), xytext=(2.52, 1.15),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(1.5, 0.78), xytext=(1.5, 2.98),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray', linestyle='--'))

    # Tools available (right side)
    ax.text(10, 7, 'Available Tools', fontsize=11, weight='bold', ha='center')

    tools = [
        ('Search', '#3498db'),
        ('Calculator', '#27ae60'),
        ('Database', '#e74c3c'),
        ('API Call', '#f39c12'),
        ('Code Exec', '#9b59b6')
    ]

    for i, (tool, color) in enumerate(tools):
        y = 6 - i * 0.7
        tool_box = FancyBboxPatch((9, y), 2, 0.5, boxstyle="round,pad=0.05",
                                 facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.7)
        ax.add_patch(tool_box)
        ax.text(10, y + 0.25, tool, fontsize=9, ha='center', color='white', weight='bold')

    # Agent executor (center bottom)
    executor = FancyBboxPatch((8.5, 0.3), 3, 1.8, boxstyle="round,pad=0.1",
                             facecolor='#f3e5f5', edgecolor='#9c27b0', linewidth=3)
    ax.add_patch(executor)
    ax.text(10, 1.9, 'Agent Executor', fontsize=11, ha='center', weight='bold')
    ax.text(10, 1.5, 'LLM (GPT-4)', fontsize=9, ha='center')
    ax.text(10, 1.2, '+ Tool Access', fontsize=9, ha='center')
    ax.text(10, 0.9, '+ Memory', fontsize=9, ha='center')
    ax.text(10, 0.6, '+ Prompt Template', fontsize=8, ha='center', style='italic')

    # Arrows from thought boxes to executor
    for y in [5.55, 3.35, 1.15]:
        ax.annotate('', xy=(8.4, 1.2), xytext=(7.5, y),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'agent_workflow.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: agent_workflow.png")

def create_lcel_pipeline():
    """LangChain Expression Language (LCEL) pipeline visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.5, 'LCEL Pipeline: Declarative Chain Composition',
            fontsize=14, weight='bold', ha='center')

    # Traditional approach (top)
    ax.text(7, 6.7, 'Traditional Approach', fontsize=11, weight='bold', ha='center', color='#e74c3c')

    trad_components = [
        ('Prompt', '#fff3cd', 0.5),
        ('LLM', '#3498db', 2.5),
        ('Parser', '#27ae60', 4.5),
        ('Store', '#e74c3c', 6.5)
    ]

    for i, (name, color, x) in enumerate(trad_components):
        box = FancyBboxPatch((x, 5.5), 1.5, 0.7, boxstyle="round,pad=0.05",
                            facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(box)
        ax.text(x + 0.75, 5.85, name, fontsize=9, ha='center', weight='bold')

        if i < len(trad_components) - 1:
            # Manual glue code
            ax.text(x + 1.6, 5.85, 'glue\ncode', fontsize=6, ha='center',
                   style='italic', color='red')
            ax.annotate('', xy=(x + 2.0, 5.85), xytext=(x + 1.52, 5.85),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # LCEL approach (bottom)
    ax.text(7, 4.5, 'LCEL Approach (chain = prompt | llm | parser | store)',
            fontsize=11, weight='bold', ha='center', color='#27ae60', family='monospace')

    lcel_components = [
        ('Prompt\nTemplate', '#fff3cd', 1),
        ('LLM\nModel', '#3498db', 3.5),
        ('Output\nParser', '#27ae60', 6),
        ('Vector\nStore', '#e74c3c', 8.5)
    ]

    for i, (name, color, x) in enumerate(lcel_components):
        box = FancyBboxPatch((x, 3), 2, 1, boxstyle="round,pad=0.08",
                            facecolor=color, edgecolor='black', linewidth=2.5, alpha=0.8)
        ax.add_patch(box)
        ax.text(x + 1, 3.5, name, fontsize=10, ha='center', weight='bold')

        if i < len(lcel_components) - 1:
            # Pipe operator
            ax.text(x + 2.2, 3.5, '|', fontsize=20, ha='center',
                   weight='bold', color='#9b59b6')
            ax.annotate('', xy=(x + 2.5, 3.5), xytext=(x + 2.05, 3.5),
                       arrowprops=dict(arrowstyle='->', lw=3, color='#9b59b6'))

    # Features boxes
    features = [
        ('Streaming\nSupport', 1, 1.5, '#e3f2fd'),
        ('Async\nExecution', 4, 1.5, '#f3e5f5'),
        ('Batch\nProcessing', 7, 1.5, '#fff9c4'),
        ('Parallel\nBranches', 10, 1.5, '#f8d7da')
    ]

    for name, x, y, color in features:
        box = FancyBboxPatch((x, y), 1.8, 0.8, boxstyle="round,pad=0.05",
                            facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + 0.9, y + 0.4, name, fontsize=8, ha='center', weight='bold')

        # Arrow pointing up to pipeline
        ax.annotate('', xy=(x + 0.9, 2.98), xytext=(x + 0.9, 2.32),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', linestyle='--'))

    # Benefits list
    benefits_box = FancyBboxPatch((0.3, 0.1), 13.4, 1, boxstyle="round,pad=0.1",
                                  facecolor='#d4edda', edgecolor='#28a745', linewidth=2)
    ax.add_patch(benefits_box)
    ax.text(7, 0.85, 'LCEL Benefits', fontsize=11, weight='bold', ha='center')

    benefits_text = [
        '✓ Type-safe composition',
        '✓ Automatic retry logic',
        '✓ Built-in observability',
        '✓ Optimized execution',
        '✓ Easy debugging'
    ]

    for i, benefit in enumerate(benefits_text):
        x = 1.5 + (i * 2.5)
        ax.text(x, 0.4, benefit, fontsize=8, ha='left')

    plt.tight_layout()
    plt.savefig(output_dir / 'lcel_pipeline.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: lcel_pipeline.png")

def create_rag_architecture():
    """RAG (Retrieval-Augmented Generation) detailed architecture"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(7, 9.5, 'RAG Architecture: Retrieval-Augmented Generation',
            fontsize=14, weight='bold', ha='center')

    # Stage 1: Document Processing (left)
    ax.text(2, 8.5, 'Stage 1: Indexing', fontsize=11, weight='bold', color='#2E86AB')

    docs_box = FancyBboxPatch((0.5, 7.2), 1.5, 0.6, boxstyle="round,pad=0.05",
                             facecolor='#e8f4f8', edgecolor='#2E86AB', linewidth=2)
    ax.add_patch(docs_box)
    ax.text(1.25, 7.5, 'Documents', fontsize=9, ha='center', weight='bold')

    # Chunking
    chunk_box = FancyBboxPatch((0.5, 6.2), 1.5, 0.6, boxstyle="round,pad=0.05",
                              facecolor='#fff3cd', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(chunk_box)
    ax.text(1.25, 6.5, 'Chunking', fontsize=9, ha='center', weight='bold')
    ax.text(1.25, 6.3, '(512 tokens)', fontsize=7, ha='center', style='italic')

    ax.annotate('', xy=(1.25, 6.18), xytext=(1.25, 7.18),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Embedding
    embed_box = FancyBboxPatch((0.5, 5.2), 1.5, 0.6, boxstyle="round,pad=0.05",
                              facecolor='#d4edda', edgecolor='#28a745', linewidth=2)
    ax.add_patch(embed_box)
    ax.text(1.25, 5.5, 'Embedding', fontsize=9, ha='center', weight='bold')
    ax.text(1.25, 5.3, 'Model', fontsize=7, ha='center', style='italic')

    ax.annotate('', xy=(1.25, 5.18), xytext=(1.25, 6.18),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Vector Store
    vdb_box = FancyBboxPatch((0.5, 3.7), 1.5, 1, boxstyle="round,pad=0.08",
                            facecolor='#f8d7da', edgecolor='#e74c3c', linewidth=2.5)
    ax.add_patch(vdb_box)
    ax.text(1.25, 4.5, 'Vector DB', fontsize=10, ha='center', weight='bold')
    ax.text(1.25, 4.2, 'Pinecone', fontsize=7, ha='center')
    ax.text(1.25, 3.95, 'ChromaDB', fontsize=7, ha='center')

    ax.annotate('', xy=(1.25, 4.68), xytext=(1.25, 5.18),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Stage 2: Retrieval (center)
    ax.text(7, 8.5, 'Stage 2: Retrieval', fontsize=11, weight='bold', color='#27ae60')

    # User query
    query_box = FancyBboxPatch((5.5, 7.2), 3, 0.6, boxstyle="round,pad=0.05",
                              facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=2)
    ax.add_patch(query_box)
    ax.text(7, 7.5, 'User Query', fontsize=10, ha='center', weight='bold')

    # Query embedding
    qembed_box = FancyBboxPatch((6, 6.2), 2, 0.6, boxstyle="round,pad=0.05",
                               facecolor='#d4edda', edgecolor='#28a745', linewidth=2)
    ax.add_patch(qembed_box)
    ax.text(7, 6.5, 'Embed Query', fontsize=9, ha='center', weight='bold')

    ax.annotate('', xy=(7, 6.18), xytext=(7, 7.18),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Similarity search
    search_box = FancyBboxPatch((5.5, 5), 3, 0.8, boxstyle="round,pad=0.05",
                               facecolor='#f3e5f5', edgecolor='#9c27b0', linewidth=2)
    ax.add_patch(search_box)
    ax.text(7, 5.6, 'Similarity Search', fontsize=9, ha='center', weight='bold')
    ax.text(7, 5.25, 'cosine(q, d) > threshold', fontsize=7, ha='center',
           family='monospace', style='italic')

    ax.annotate('', xy=(7, 4.98), xytext=(7, 6.18),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Arrow from vector DB to search
    ax.annotate('', xy=(5.48, 5.4), xytext=(2.02, 4.2),
               arrowprops=dict(arrowstyle='->', lw=2, color='#e74c3c'))
    ax.text(3.5, 5, 'Retrieve\nTop-K', fontsize=7, ha='center', color='#e74c3c', weight='bold')

    # Retrieved documents
    retrieved_box = FancyBboxPatch((5.5, 3.7), 3, 1, boxstyle="round,pad=0.08",
                                  facecolor='#fff3cd', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(retrieved_box)
    ax.text(7, 4.5, 'Retrieved Context', fontsize=10, ha='center', weight='bold')
    ax.text(7, 4.2, 'Doc 1: relevance 0.95', fontsize=7, ha='center', family='monospace')
    ax.text(7, 3.95, 'Doc 2: relevance 0.89', fontsize=7, ha='center', family='monospace')

    ax.annotate('', xy=(7, 4.68), xytext=(7, 4.98),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Stage 3: Generation (right)
    ax.text(11.5, 8.5, 'Stage 3: Generation', fontsize=11, weight='bold', color='#e74c3c')

    # Prompt construction
    prompt_box = FancyBboxPatch((10, 7.2), 3, 0.6, boxstyle="round,pad=0.05",
                               facecolor='#fff9c4', edgecolor='#fbc02d', linewidth=2)
    ax.add_patch(prompt_box)
    ax.text(11.5, 7.5, 'Build Prompt', fontsize=9, ha='center', weight='bold')

    # Arrows to prompt
    ax.annotate('', xy=(9.98, 7.5), xytext=(8.52, 7.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.text(9.2, 7.7, 'query', fontsize=7, ha='center', style='italic')

    ax.annotate('', xy=(10.5, 7.18), xytext=(7, 4.68),
               arrowprops=dict(arrowstyle='->', lw=2, color='#f39c12'))
    ax.text(8.5, 6, 'context', fontsize=7, ha='center', color='#f39c12', weight='bold')

    # LLM
    llm_box = FancyBboxPatch((10, 5.7), 3, 1.2, boxstyle="round,pad=0.1",
                            facecolor='#3498db', edgecolor='black', linewidth=2.5, alpha=0.8)
    ax.add_patch(llm_box)
    ax.text(11.5, 6.6, 'LLM', fontsize=11, ha='center', weight='bold', color='white')
    ax.text(11.5, 6.2, 'GPT-4 / Claude', fontsize=8, ha='center', color='white')

    ax.annotate('', xy=(11.5, 6.88), xytext=(11.5, 7.18),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

    # Answer
    answer_box = FancyBboxPatch((10, 4.2), 3, 1, boxstyle="round,pad=0.08",
                               facecolor='#d4edda', edgecolor='#28a745', linewidth=2.5)
    ax.add_patch(answer_box)
    ax.text(11.5, 5, 'Generated Answer', fontsize=10, ha='center', weight='bold')
    ax.text(11.5, 4.7, 'Factually grounded', fontsize=8, ha='center', style='italic')
    ax.text(11.5, 4.4, 'in retrieved context', fontsize=8, ha='center', style='italic')

    ax.annotate('', xy=(11.5, 5.18), xytext=(11.5, 5.68),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

    # Key metrics (bottom)
    metrics_box = FancyBboxPatch((0.5, 0.3), 13, 3, boxstyle="round,pad=0.1",
                                 facecolor='#f5f5f5', edgecolor='gray', linewidth=2)
    ax.add_patch(metrics_box)

    ax.text(7, 3, 'RAG Performance Factors', fontsize=11, weight='bold', ha='center')

    factors = [
        ('Chunk Size', 'Smaller: precise\nLarger: context', 1.5),
        ('Top-K', 'More: comprehensive\nFewer: focused', 4.5),
        ('Embedding\nModel', 'Quality affects\nretrieval accuracy', 7.5),
        ('Reranking', 'Improves relevance\nof final results', 10.5)
    ]

    for title, desc, x in factors:
        ax.text(x, 2.3, title, fontsize=9, ha='center', weight='bold')
        ax.text(x, 1.8, desc, fontsize=7, ha='center', style='italic')

        # Mini bars showing trade-off
        bar_y = [1.3, 1.0]
        bar_height = 0.2
        colors = ['#3498db', '#e74c3c']
        for i, (y, color) in enumerate(zip(bar_y, colors)):
            rect = Rectangle((x - 0.4, y), 0.8, bar_height, facecolor=color,
                           edgecolor='black', linewidth=0.5, alpha=0.6)
            ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(output_dir / 'rag_architecture.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: rag_architecture.png")

def main():
    """Generate all LangChain visualizations"""
    print("=" * 60)
    print("Generating LangChain Visualizations")
    print("=" * 60)

    create_langchain_architecture()
    create_chain_types()
    create_memory_types()
    create_agent_workflow()
    create_lcel_pipeline()
    create_rag_architecture()

    print("=" * 60)
    print(f"✓ All visualizations saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
