"""
LangGraph Visualization Script
Generates publication-quality visualizations for LangGraph section of LLM Study Material
Total: 8 visualizations covering state management, graphs, agents, and workflows
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle, Polygon
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
output_dir = Path(__file__).parent.parent / 'images' / 'langgraph'
output_dir.mkdir(parents=True, exist_ok=True)

def create_state_management():
    """State management with reducers visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 1. State structure diagram
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.text(5, 9.5, 'State Structure (TypedDict)', fontsize=12, weight='bold', ha='center')

    # State box
    state_box = FancyBboxPatch((1, 3), 8, 5.5, boxstyle="round,pad=0.1",
                               facecolor='#e8f4f8', edgecolor='#2E86AB', linewidth=2.5)
    ax1.add_patch(state_box)
    ax1.text(5, 8, 'AgentState', fontsize=11, weight='bold', ha='center')

    # Fields
    fields = [
        ('messages: Annotated[list, add]', '#d4edda', 'Append reducer'),
        ('current_step: str', '#fff3cd', 'Replace value'),
        ('iteration_count: int', '#f8d7da', 'Replace value'),
        ('results: dict', '#e3f2fd', 'Replace value')
    ]

    y = 7
    for field, color, desc in fields:
        field_box = FancyBboxPatch((1.5, y-0.6), 7, 0.5, boxstyle="round,pad=0.03",
                                   facecolor=color, edgecolor='black', linewidth=1)
        ax1.add_patch(field_box)
        ax1.text(2, y-0.35, field, fontsize=8, family='monospace', weight='bold')
        ax1.text(7.5, y-0.35, desc, fontsize=7, ha='right', style='italic')
        y -= 0.8

    ax1.text(5, 3.5, 'State = Python Dictionary', fontsize=9, ha='center', style='italic')
    ax1.text(5, 3.1, 'Updated by nodes, passed through graph', fontsize=8, ha='center')

    # 2. Reducer comparison
    reducers = ['Replace', 'Append\n(add)', 'Custom\n(lambda)']
    old_val = [5, 3, 2]
    new_val = [8, 2, 4]
    result = [8, 5, 4]  # Replace=8, Append=3+2=5, Custom=max([2,4])=4

    x = np.arange(len(reducers))
    width = 0.25

    ax2.bar(x - width, old_val, width, label='Old Value', color='#95a5a6', alpha=0.8)
    ax2.bar(x, new_val, width, label='New Value', color='#3498db', alpha=0.8)
    ax2.bar(x + width, result, width, label='Result', color='#27ae60', alpha=0.8)

    ax2.set_ylabel('Value', fontsize=11, weight='bold')
    ax2.set_title('Reducer Behavior Comparison', fontsize=12, weight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(reducers, fontsize=9)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 3. State update flow
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.text(5, 9.5, 'State Update Flow', fontsize=12, weight='bold', ha='center')

    # Timeline
    steps = ['S₀\nInitial', 'Node 1', 'S₁', 'Node 2', 'S₂', 'Node 3', 'S₃\nFinal']
    colors_step = ['#95a5a6', '#3498db', '#95a5a6', '#3498db', '#95a5a6', '#3498db', '#27ae60']

    for i, (step, color) in enumerate(zip(steps, colors_step)):
        x = 1.5 + i * 1.2
        y = 5
        if 'Node' in step:
            box = FancyBboxPatch((x-0.4, y-0.4), 0.8, 0.8, boxstyle="round,pad=0.05",
                                facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.7)
        else:
            box = Circle((x, y), 0.4, facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.7)
        ax3.add_patch(box)
        ax3.text(x, y, step.split('\n')[0], fontsize=7, ha='center', va='center', weight='bold')
        if '\n' in step:
            ax3.text(x, y-0.7, step.split('\n')[1], fontsize=6, ha='center', style='italic')

        if i < len(steps) - 1:
            ax3.annotate('', xy=(x+0.45, y), xytext=(x+0.4, y),
                        arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax3.text(5, 3.5, 'Each node updates state: S_{t+1} = S_t ⊕ ΔS',
            fontsize=9, ha='center', family='serif', style='italic')
    ax3.text(5, 2.8, 'Reducers control HOW fields merge', fontsize=8, ha='center')

    # 4. Memory growth
    steps_count = np.array([0, 5, 10, 15, 20, 25, 30])

    # Replace: constant size
    replace_size = np.ones_like(steps_count) * 10

    # Append: linear growth
    append_size = 10 + steps_count * 2

    # Custom (with pruning): bounded growth
    custom_size = 10 + np.minimum(steps_count * 2, 30)

    ax4.plot(steps_count, replace_size, 'o-', linewidth=2.5, markersize=7,
            label='Replace (constant)', color='#27ae60')
    ax4.plot(steps_count, append_size, 's-', linewidth=2.5, markersize=7,
            label='Append (linear)', color='#e74c3c')
    ax4.plot(steps_count, custom_size, '^-', linewidth=2.5, markersize=7,
            label='Custom with pruning', color='#3498db')

    ax4.set_xlabel('Number of Steps', fontsize=11, weight='bold')
    ax4.set_ylabel('State Size (arbitrary units)', fontsize=11, weight='bold')
    ax4.set_title('State Memory Growth Over Time', fontsize=12, weight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'state_management.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: state_management.png")

def create_graph_execution():
    """Graph construction and execution flow"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(7, 9.5, 'LangGraph: Graph Construction & Execution Flow',
            fontsize=14, weight='bold', ha='center')

    # Graph definition (left side)
    ax.text(2.5, 8.5, 'Graph Definition', fontsize=11, weight='bold', color='#2E86AB')

    # StateGraph creation
    sg_box = FancyBboxPatch((0.5, 7.5), 4, 0.6, boxstyle="round,pad=0.05",
                            facecolor='#e8f4f8', edgecolor='#2E86AB', linewidth=2)
    ax.add_patch(sg_box)
    ax.text(2.5, 7.8, 'workflow = StateGraph(AgentState)', fontsize=8,
           ha='center', family='monospace', weight='bold')

    # Add nodes
    nodes = [
        ('agent', '#3498db', 6.7),
        ('tools', '#27ae60', 5.9),
        ('summarize', '#f39c12', 5.1)
    ]

    for name, color, y in nodes:
        node_box = FancyBboxPatch((0.7, y), 3.6, 0.5, boxstyle="round,pad=0.03",
                                  facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.6)
        ax.add_patch(node_box)
        ax.text(1, y+0.25, f'add_node("{name}")', fontsize=7, family='monospace',
               color='white', weight='bold', va='center')

    # Add edges
    edge_box = FancyBboxPatch((0.7, 4.3), 3.6, 0.5, boxstyle="round,pad=0.03",
                              facecolor='#9b59b6', edgecolor='black', linewidth=1.5, alpha=0.6)
    ax.add_patch(edge_box)
    ax.text(1, 4.55, 'add_conditional_edges(...)', fontsize=7, family='monospace',
           color='white', weight='bold', va='center')

    # Entry point
    entry_box = FancyBboxPatch((0.7, 3.5), 3.6, 0.5, boxstyle="round,pad=0.03",
                               facecolor='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.6)
    ax.add_patch(entry_box)
    ax.text(1, 3.75, 'set_entry_point("agent")', fontsize=7, family='monospace',
           color='white', weight='bold', va='center')

    # Compile
    compile_box = FancyBboxPatch((0.7, 2.7), 3.6, 0.5, boxstyle="round,pad=0.05",
                                 facecolor='#2c3e50', edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(compile_box)
    ax.text(1, 2.95, 'app = workflow.compile()', fontsize=8, family='monospace',
           color='white', weight='bold', va='center')

    # Arrow to execution
    ax.annotate('', xy=(5.8, 5), xytext=(4.5, 5),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax.text(5.1, 5.3, 'Execute', fontsize=9, weight='bold')

    # Execution flow (right side)
    ax.text(10, 8.5, 'Execution Flow', fontsize=11, weight='bold', color='#27ae60')

    # Start
    start = Circle((10, 7.5), 0.4, facecolor='#27ae60', edgecolor='black', linewidth=2)
    ax.add_patch(start)
    ax.text(10, 7.5, 'START', fontsize=7, ha='center', va='center', weight='bold', color='white')

    # Agent node
    agent_exec = FancyBboxPatch((8.5, 6.2), 3, 0.8, boxstyle="round,pad=0.08",
                                facecolor='#3498db', edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(agent_exec)
    ax.text(10, 6.8, 'Agent Node', fontsize=9, ha='center', weight='bold', color='white')
    ax.text(10, 6.4, 'Reasoning', fontsize=7, ha='center', color='white', style='italic')

    ax.annotate('', xy=(10, 6.18), xytext=(10, 6.9),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

    # Conditional router
    router = Polygon([[10, 5.5], [11.2, 5], [10, 4.5], [8.8, 5]],
                     facecolor='#f39c12', edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(router)
    ax.text(10, 5, 'Route?', fontsize=8, ha='center', weight='bold')

    ax.annotate('', xy=(10, 5.5), xytext=(10, 6.18),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

    # Tools node
    tools_exec = FancyBboxPatch((6, 3.2), 2.5, 0.8, boxstyle="round,pad=0.08",
                                facecolor='#27ae60', edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(tools_exec)
    ax.text(7.25, 3.8, 'Tools', fontsize=9, ha='center', weight='bold', color='white')
    ax.text(7.25, 3.4, 'Execute', fontsize=7, ha='center', color='white', style='italic')

    ax.annotate('', xy=(7.8, 3.6), xytext=(8.8, 4.8),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(7.5, 4.3, 'tools', fontsize=7, style='italic')

    # Loop back
    ax.annotate('', xy=(8.8, 6.6), xytext=(8.2, 3.8),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray', linestyle='--',
                             connectionstyle="arc3,rad=.5"))
    ax.text(7, 5.2, 'loop', fontsize=7, color='gray', style='italic')

    # Summarize node
    summ_exec = FancyBboxPatch((9.5, 2), 2.5, 0.8, boxstyle="round,pad=0.08",
                               facecolor='#9b59b6', edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(summ_exec)
    ax.text(10.75, 2.6, 'Summarize', fontsize=9, ha='center', weight='bold', color='white')
    ax.text(10.75, 2.2, 'Final', fontsize=7, ha='center', color='white', style='italic')

    ax.annotate('', xy=(10.5, 2.82), xytext=(10.5, 4.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(10.8, 3.5, 'sum', fontsize=7, style='italic')

    # END
    end = Circle((10.75, 0.8), 0.4, facecolor='#e74c3c', edgecolor='black', linewidth=2)
    ax.add_patch(end)
    ax.text(10.75, 0.8, 'END', fontsize=7, ha='center', va='center', weight='bold', color='white')

    ax.annotate('', xy=(10.75, 1.22), xytext=(10.75, 1.98),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

    # State updates annotation
    state_updates = FancyBboxPatch((5.5, 0.3), 4, 0.4, boxstyle="round,pad=0.05",
                                   facecolor='#fff3cd', edgecolor='#f39c12', linewidth=1.5)
    ax.add_patch(state_updates)
    ax.text(7.5, 0.5, 'State updated at each node: S_{t+1} = S_t ⊕ ΔS',
           fontsize=8, ha='center', family='serif', style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / 'graph_execution.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: graph_execution.png")

def create_conditional_routing():
    """Conditional edges and routing decision tree"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    ax.text(7, 8.5, 'Conditional Routing: should_continue() Function',
            fontsize=14, weight='bold', ha='center')

    # Current state
    state_box = FancyBboxPatch((5.5, 7), 3, 0.7, boxstyle="round,pad=0.05",
                               facecolor='#e8f4f8', edgecolor='#2E86AB', linewidth=2)
    ax.add_patch(state_box)
    ax.text(7, 7.35, 'Current State (S_t)', fontsize=10, ha='center', weight='bold')

    # Routing function
    router_box = FancyBboxPatch((5, 5.5), 4, 1, boxstyle="round,pad=0.08",
                                facecolor='#f39c12', edgecolor='black', linewidth=2.5, alpha=0.8)
    ax.add_patch(router_box)
    ax.text(7, 6.2, 'should_continue(state)', fontsize=10, ha='center', weight='bold')
    ax.text(7, 5.85, 'Decision Logic', fontsize=8, ha='center', style='italic')

    ax.annotate('', xy=(7, 6.48), xytext=(7, 6.98),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

    # Decision checks
    checks = [
        ('Has tool_calls?', 2, 3.8, '#3498db', 'tools'),
        ('iteration > 10?', 7, 3.8, '#e74c3c', 'end'),
        ('Task complete?', 12, 3.8, '#27ae60', 'agent')
    ]

    for label, x, y, color, result in checks:
        # Check box
        check = FancyBboxPatch((x-1, y), 2, 0.8, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.6)
        ax.add_patch(check)
        ax.text(x, y+0.55, label, fontsize=8, ha='center', weight='bold', color='white')
        ax.text(x, y+0.2, f'→ "{result}"', fontsize=7, ha='center', color='white', family='monospace')

        # Arrow from router
        ax.annotate('', xy=(x, y+0.82), xytext=(7, 5.48),
                   arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

    # Outcomes
    outcomes = [
        ('TOOLS\nNode', 2, 2, '#3498db', 'Execute tools'),
        ('END\nTerminate', 7, 2, '#e74c3c', 'Stop execution'),
        ('AGENT\nNode', 12, 2, '#27ae60', 'Continue reasoning')
    ]

    for label, x, y, color, desc in outcomes:
        outcome_box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8, boxstyle="round,pad=0.08",
                                     facecolor=color, edgecolor='black', linewidth=2, alpha=0.9)
        ax.add_patch(outcome_box)
        ax.text(x, y+0.1, label.split('\n')[0], fontsize=9, ha='center', weight='bold', color='white')
        ax.text(x, y-0.15, label.split('\n')[1], fontsize=7, ha='center', color='white', style='italic')
        ax.text(x, y-0.7, desc, fontsize=7, ha='center')

        # Arrow from check to outcome
        ax.annotate('', xy=(x, y+0.42), xytext=(x, 3.78),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Loop annotation
    ax.annotate('', xy=(12.5, 6), xytext=(12.5, 2.4),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='#27ae60', linestyle='--',
                             connectionstyle="arc3,rad=.8"))
    ax.text(13.2, 4, 'Loop', fontsize=9, color='#27ae60', weight='bold', rotation=-90, va='center')

    # Code example
    code_box = FancyBboxPatch((0.5, 0.3), 13, 1.2, boxstyle="round,pad=0.1",
                              facecolor='#2c3e50', edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(code_box)

    code = [
        'def should_continue(state):',
        '    if state["messages"][-1].tool_calls: return "tools"',
        '    if state["iteration_count"] > 10: return "end"',
        '    return "agent"  # Continue reasoning'
    ]

    y_code = 1.2
    for line in code:
        ax.text(1, y_code, line, fontsize=7, family='monospace', color='white')
        y_code -= 0.2

    plt.tight_layout()
    plt.savefig(output_dir / 'conditional_routing.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: conditional_routing.png")

def create_multi_agent():
    """Multi-agent supervisor architecture"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(7, 9.5, 'Multi-Agent System: Supervisor Pattern',
            fontsize=14, weight='bold', ha='center')

    # Supervisor (center)
    supervisor = Circle((7, 6), 1, facecolor='#f39c12', edgecolor='black', linewidth=3, alpha=0.9)
    ax.add_patch(supervisor)
    ax.text(7, 6.3, 'SUPERVISOR', fontsize=10, ha='center', weight='bold')
    ax.text(7, 5.7, 'Orchestrator', fontsize=8, ha='center', style='italic')

    # Specialized agents (around supervisor)
    agents = [
        ('Researcher', 2, 6, '#3498db', 'Search & Gather'),
        ('Writer', 7, 9, '#27ae60', 'Generate Content'),
        ('Critic', 12, 6, '#e74c3c', 'Review & Improve'),
        ('Executor', 7, 3, '#9b59b6', 'Execute Actions')
    ]

    for name, x, y, color, role in agents:
        agent_circle = Circle((x, y), 0.8, facecolor=color, edgecolor='black',
                             linewidth=2, alpha=0.8)
        ax.add_patch(agent_circle)
        ax.text(x, y+0.15, name, fontsize=9, ha='center', weight='bold', color='white')
        ax.text(x, y-0.15, role, fontsize=7, ha='center', color='white', style='italic')

        # Arrows to/from supervisor
        # Task assignment (supervisor → agent)
        dx_out = (x - 7) * 0.3
        dy_out = (y - 6) * 0.3
        ax.annotate('', xy=(x - dx_out*0.5, y - dy_out*0.5),
                   xytext=(7 + dx_out*0.8, 6 + dy_out*0.8),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        ax.text(7 + dx_out*1.3, 6 + dy_out*1.3, 'task', fontsize=6, ha='center',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        # Result return (agent → supervisor)
        ax.annotate('', xy=(7 + dx_out*0.8, 6 + dy_out*0.8),
                   xytext=(x - dx_out*0.5, y - dy_out*0.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='gray', linestyle='--'))
        ax.text(x - dx_out*1.3, y - dy_out*1.3, 'result', fontsize=6, ha='center',
               color='gray', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    # User input
    user_box = FancyBboxPatch((5.5, 0.3), 3, 0.6, boxstyle="round,pad=0.05",
                              facecolor='#e8f4f8', edgecolor='#2E86AB', linewidth=2)
    ax.add_patch(user_box)
    ax.text(7, 0.6, 'User Request', fontsize=9, ha='center', weight='bold')

    ax.annotate('', xy=(7, 4.9), xytext=(7, 0.92),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='#2E86AB'))

    # Final output
    output_box = FancyBboxPatch((10.5, 0.3), 3, 0.6, boxstyle="round,pad=0.05",
                                facecolor='#d4edda', edgecolor='#28a745', linewidth=2)
    ax.add_patch(output_box)
    ax.text(12, 0.6, 'Final Answer', fontsize=9, ha='center', weight='bold')

    ax.annotate('', xy=(10.4, 0.6), xytext=(8.1, 5.5),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='#28a745'))

    # State sharing
    state_box = FancyBboxPatch((0.3, 4.5), 2.5, 3.5, boxstyle="round,pad=0.1",
                               facecolor='#fff3cd', edgecolor='#f39c12', linewidth=2, alpha=0.5)
    ax.add_patch(state_box)
    ax.text(1.55, 7.7, 'Shared State', fontsize=10, ha='center', weight='bold')
    ax.text(1.55, 7.3, '• messages', fontsize=8, ha='center')
    ax.text(1.55, 6.9, '• next_agent', fontsize=8, ha='center')
    ax.text(1.55, 6.5, '• task_queue', fontsize=8, ha='center')
    ax.text(1.55, 6.1, '• results', fontsize=8, ha='center')
    ax.text(1.55, 5.5, 'All agents\nread/write', fontsize=7, ha='center', style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / 'multi_agent.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: multi_agent.png")

def create_human_in_loop():
    """Human-in-the-loop checkpoint system"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    ax.text(7, 8.5, 'Human-in-the-Loop: Checkpointing & Interrupts',
            fontsize=14, weight='bold', ha='center')

    # Timeline
    steps = [
        ('Start', 1, '#27ae60'),
        ('Node 1', 2.5, '#3498db'),
        ('CP₁', 4, '#95a5a6'),
        ('Node 2', 5.5, '#3498db'),
        ('CP₂', 7, '#95a5a6'),
        ('⏸️ Pause', 8.5, '#f39c12'),
        ('Human\nReview', 10, '#e74c3c'),
        ('Resume', 11.5, '#27ae60'),
        ('End', 13, '#2c3e50')
    ]

    y_timeline = 6
    for label, x, color in steps:
        if 'CP' in label or 'Pause' in label:
            shape = FancyBboxPatch((x-0.3, y_timeline-0.3), 0.6, 0.6,
                                  boxstyle="round,pad=0.05",
                                  facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.7)
        elif 'Human' in label:
            shape = FancyBboxPatch((x-0.4, y_timeline-0.4), 0.8, 0.8,
                                  boxstyle="round,pad=0.08",
                                  facecolor=color, edgecolor='black', linewidth=2, alpha=0.9)
        else:
            shape = Circle((x, y_timeline), 0.35, facecolor=color,
                          edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.add_patch(shape)
        ax.text(x, y_timeline, label.split('\n')[0], fontsize=7 if len(label)<8 else 6,
               ha='center', va='center', weight='bold')
        if '\n' in label:
            ax.text(x, y_timeline-0.2, label.split('\n')[1], fontsize=6,
                   ha='center', va='center')

    # Timeline line
    ax.plot([1, 13], [y_timeline, y_timeline], 'k-', linewidth=1, alpha=0.3, zorder=0)

    # Checkpoint saving
    ax.text(4, 5, 'Auto-save', fontsize=7, ha='center', style='italic')
    ax.annotate('', xy=(4, 5.3), xytext=(4, 5.65),
               arrowprops=dict(arrowstyle='->', lw=1, color='gray', linestyle='--'))

    ax.text(7, 5, 'Auto-save', fontsize=7, ha='center', style='italic')
    ax.annotate('', xy=(7, 5.3), xytext=(7, 5.65),
               arrowprops=dict(arrowstyle='->', lw=1, color='gray', linestyle='--'))

    # Interrupt detail
    interrupt_box = FancyBboxPatch((7.5, 3.5), 3.5, 2, boxstyle="round,pad=0.1",
                                   facecolor='#fff3cd', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(interrupt_box)
    ax.text(9.25, 5.2, 'Interrupt Point', fontsize=10, ha='center', weight='bold')
    ax.text(9.25, 4.8, '1. Save current state', fontsize=7, ha='center')
    ax.text(9.25, 4.5, '2. Show planned action', fontsize=7, ha='center')
    ax.text(9.25, 4.2, '3. Wait for approval', fontsize=7, ha='center')
    ax.text(9.25, 3.9, '4. Resume or modify', fontsize=7, ha='center')

    # Human actions
    human_actions = FancyBboxPatch((9, 1.5), 4, 1.5, boxstyle="round,pad=0.1",
                                   facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=2)
    ax.add_patch(human_actions)
    ax.text(11, 2.7, 'Human Options', fontsize=10, ha='center', weight='bold')
    ax.text(11, 2.35, '✓ Approve: Continue', fontsize=8, ha='center')
    ax.text(11, 2.05, '✗ Reject: Stop', fontsize=8, ha='center')
    ax.text(11, 1.75, '✎ Modify: Edit and resume', fontsize=8, ha='center')

    ax.annotate('', xy=(10, 2.98), xytext=(10, 3.48),
               arrowprops=dict(arrowstyle='<->', lw=2, color='#1976d2'))

    # Thread persistence
    thread_box = FancyBboxPatch((0.5, 1.5), 3.5, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#f8d7da', edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(thread_box)
    ax.text(2.25, 2.7, 'Thread Storage', fontsize=10, ha='center', weight='bold')
    ax.text(2.25, 2.35, 'thread_id: "user_123"', fontsize=7, ha='center', family='monospace')
    ax.text(2.25, 2.05, 'Checkpoints: CP₁, CP₂', fontsize=7, ha='center')
    ax.text(2.25, 1.75, 'Resume anytime', fontsize=7, ha='center', style='italic')

    # Benefits
    benefits_box = FancyBboxPatch((0.5, 0.2), 13, 0.8, boxstyle="round,pad=0.05",
                                  facecolor='#d4edda', edgecolor='#28a745', linewidth=2)
    ax.add_patch(benefits_box)
    ax.text(7, 0.8, 'Benefits: Safety (human approval), Flexibility (resume later), ' +
           'Transparency (see planned actions)', fontsize=8, ha='center', style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / 'human_in_loop.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: human_in_loop.png")

def create_react_loop():
    """ReAct agent reasoning-action-observation loop"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(7, 9.5, 'ReAct Agent: Reasoning + Action Loop',
            fontsize=14, weight='bold', ha='center')

    # Central cycle
    center_x, center_y = 7, 5.5
    radius = 2.5

    phases = [
        ('Thought', 90, '#3498db', 'Reason about\nnext step'),
        ('Action', 330, '#27ae60', 'Execute tool\nor query'),
        ('Observation', 210, '#f39c12', 'Process\nresult')
    ]

    for label, angle, color, desc in phases:
        rad = np.radians(angle)
        x = center_x + radius * np.cos(rad)
        y = center_y + radius * np.sin(rad)

        phase_box = FancyBboxPatch((x-0.7, y-0.5), 1.4, 1, boxstyle="round,pad=0.1",
                                   facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(phase_box)
        ax.text(x, y+0.15, label, fontsize=10, ha='center', weight='bold', color='white')
        ax.text(x, y-0.25, desc, fontsize=7, ha='center', color='white', style='italic')

    # Circular arrows
    for i in range(3):
        start_angle = phases[i][1]
        end_angle = phases[(i+1)%3][1]

        start_rad = np.radians(start_angle - 35)
        end_rad = np.radians(end_angle + 35)

        start_x = center_x + radius * np.cos(start_rad)
        start_y = center_y + radius * np.sin(start_rad)
        end_x = center_x + radius * np.cos(end_rad)
        end_y = center_y + radius * np.sin(end_rad)

        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='black',
                                 connectionstyle="arc3,rad=.3"))

    # Example walkthrough
    example_box = FancyBboxPatch((0.3, 0.3), 6, 3, boxstyle="round,pad=0.1",
                                 facecolor='#e8f4f8', edgecolor='#2E86AB', linewidth=2)
    ax.add_patch(example_box)
    ax.text(3.3, 3.1, 'Example: "What is 25 × 37?"', fontsize=9, ha='center', weight='bold')

    steps = [
        ('Thought₁', 'Need calculator for multiplication', 2.7),
        ('Action₁', 'calculator(25 * 37)', 2.35),
        ('Obs₁', 'Result: 925', 2.0),
        ('Thought₂', 'Have the answer', 1.65),
        ('Action₂', 'final_answer(925)', 1.3),
        ('Done', 'Answer: 925 ✓', 0.85)
    ]

    for label, text, y_pos in steps:
        color = '#3498db' if 'Thought' in label else ('#27ae60' if 'Action' in label else '#f39c12')
        if 'Done' in label:
            color = '#2c3e50'
        ax.text(0.8, y_pos, f'{label}:', fontsize=8, weight='bold', color=color)
        ax.text(2, y_pos, text, fontsize=7, style='italic')

    # Tools available
    tools_box = FancyBboxPatch((7.5, 0.3), 6, 3, boxstyle="round,pad=0.1",
                               facecolor='#fff3cd', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(tools_box)
    ax.text(10.5, 3.1, 'Available Tools', fontsize=9, ha='center', weight='bold')

    tools = [
        ('calculator', 'Math operations', '#27ae60'),
        ('search', 'Web search', '#3498db'),
        ('database', 'Query DB', '#9b59b6'),
        ('file_read', 'Read files', '#e74c3c')
    ]

    y_start = 2.6
    for i, (tool, desc, color) in enumerate(tools):
        y_pos = y_start - i*0.5
        tool_badge = FancyBboxPatch((8, y_pos-0.15), 1.2, 0.35, boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor='black', linewidth=1, alpha=0.7)
        ax.add_patch(tool_badge)
        ax.text(8.6, y_pos, tool, fontsize=7, ha='center', weight='bold', color='white')
        ax.text(10.8, y_pos, desc, fontsize=7, ha='left', style='italic')

    # Key insight
    insight_box = FancyBboxPatch((0.3, 8.2), 5, 1, boxstyle="round,pad=0.05",
                                 facecolor='#d4edda', edgecolor='#28a745', linewidth=2)
    ax.add_patch(insight_box)
    ax.text(2.8, 8.9, 'Key: Interleaved reasoning & actions', fontsize=8, ha='center', weight='bold')
    ax.text(2.8, 8.5, 'vs. Plan-Execute: Separate phases', fontsize=7, ha='center', style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / 'react_loop.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: react_loop.png")

def create_plan_execute():
    """Plan-and-Execute architecture workflow"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(7, 9.5, 'Plan-and-Execute Architecture',
            fontsize=14, weight='bold', ha='center')

    # Planning phase (left)
    plan_box = FancyBboxPatch((0.5, 3), 5.5, 5.5, boxstyle="round,pad=0.1",
                              facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=3)
    ax.add_patch(plan_box)
    ax.text(3.25, 8.3, 'Phase 1: PLANNING', fontsize=11, ha='center', weight='bold')

    # Planner LLM
    planner = FancyBboxPatch((1, 7), 4.5, 0.8, boxstyle="round,pad=0.05",
                             facecolor='#1976d2', edgecolor='black', linewidth=2)
    ax.add_patch(planner)
    ax.text(3.25, 7.4, 'Planner LLM', fontsize=9, ha='center', weight='bold', color='white')

    # Generated plan
    plan_steps = FancyBboxPatch((1, 4.5), 4.5, 2, boxstyle="round,pad=0.05",
                                facecolor='white', edgecolor='#1976d2', linewidth=1.5)
    ax.add_patch(plan_steps)
    ax.text(3.25, 6.3, 'Generated Plan:', fontsize=8, ha='center', weight='bold')
    steps = ['1. Search for information', '2. Analyze results', '3. Synthesize answer']
    for i, step in enumerate(steps):
        ax.text(3.25, 5.9 - i*0.4, step, fontsize=7, ha='center', family='monospace')

    ax.annotate('', xy=(3.25, 6.5), xytext=(3.25, 6.98),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Input
    input_box = FancyBboxPatch((1.5, 3.3), 3.5, 0.6, boxstyle="round,pad=0.05",
                               facecolor='#fff3cd', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(input_box)
    ax.text(3.25, 3.6, 'User Query', fontsize=8, ha='center', weight='bold')

    ax.annotate('', xy=(3.25, 6.98), xytext=(3.25, 3.92),
               arrowprops=dict(arrowstyle='->', lw=2, color='#f39c12'))

    # Execution phase (right)
    exec_box = FancyBboxPatch((8, 3), 5.5, 5.5, boxstyle="round,pad=0.1",
                              facecolor='#f3e5f5', edgecolor='#7b1fa2', linewidth=3)
    ax.add_patch(exec_box)
    ax.text(10.75, 8.3, 'Phase 2: EXECUTION', fontsize=11, ha='center', weight='bold')

    # Sequential execution
    exec_steps = [
        ('Step 1', 7.5, '#27ae60', 'search("topic")'),
        ('Step 2', 6.5, '#3498db', 'analyze(results)'),
        ('Step 3', 5.5, '#9b59b6', 'synthesize(data)')
    ]

    for label, y_pos, color, action in exec_steps:
        step_box = FancyBboxPatch((8.5, y_pos-0.3), 4.5, 0.6, boxstyle="round,pad=0.05",
                                  facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.add_patch(step_box)
        ax.text(9, y_pos, label, fontsize=8, ha='left', weight='bold', color='white')
        ax.text(10.75, y_pos, action, fontsize=7, ha='center', family='monospace', color='white')

        if y_pos > 5.5:
            ax.annotate('', xy=(10.75, y_pos - 0.32), xytext=(10.75, y_pos - 0.68),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Output
    output_box = FancyBboxPatch((9, 3.8), 3.5, 0.6, boxstyle="round,pad=0.05",
                                facecolor='#d4edda', edgecolor='#28a745', linewidth=2)
    ax.add_patch(output_box)
    ax.text(10.75, 4.1, 'Final Answer', fontsize=8, ha='center', weight='bold')

    ax.annotate('', xy=(10.75, 4.4), xytext=(10.75, 5.18),
               arrowprops=dict(arrowstyle='->', lw=2, color='#28a745'))

    # Arrow between phases
    ax.annotate('', xy=(7.9, 6), xytext=(6.1, 6),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax.text(7, 6.3, 'Pass Plan', fontsize=8, ha='center', weight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # State tracking
    state_box = FancyBboxPatch((8.5, 1), 4.5, 1.5, boxstyle="round,pad=0.05",
                               facecolor='#fff3cd', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(state_box)
    ax.text(10.75, 2.3, 'Execution State', fontsize=8, ha='center', weight='bold')
    ax.text(10.75, 2, 'plan: [step1, step2, step3]', fontsize=6, ha='center', family='monospace')
    ax.text(10.75, 1.7, 'past_steps: [...completed...]', fontsize=6, ha='center', family='monospace')
    ax.text(10.75, 1.4, 'results: [...outputs...]', fontsize=6, ha='center', family='monospace')

    # Comparison
    compare_box = FancyBboxPatch((0.5, 0.2), 13, 0.6, boxstyle="round,pad=0.05",
                                 facecolor='#e8f4f8', edgecolor='#2E86AB', linewidth=2)
    ax.add_patch(compare_box)
    ax.text(7, 0.65, 'vs. ReAct: Plan-Execute separates planning (upfront) from execution (sequential)',
           fontsize=7, ha='center', style='italic')
    ax.text(7, 0.35, 'ReAct interleaves reasoning and actions dynamically',
           fontsize=7, ha='center', style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / 'plan_execute.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: plan_execute.png")

def create_checkpoint_timeline():
    """Checkpoint timeline with time-travel branching"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    ax.text(7, 8.5, 'Checkpoint System: Time-Travel & Branching',
            fontsize=14, weight='bold', ha='center')

    # Main timeline
    y_main = 6
    checkpoints = [(i, f'CP{i}', '#3498db') for i in range(11)]

    ax.plot([0.5, 13.5], [y_main, y_main], 'k-', linewidth=2, alpha=0.3, zorder=0)

    for i, label, color in checkpoints:
        x = 0.5 + i * 1.2
        circle = Circle((x, y_main), 0.25, facecolor=color,
                       edgecolor='black', linewidth=1.5, alpha=0.8, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y_main-0.6, f'{i}', fontsize=7, ha='center', weight='bold')

        if i == 5:
            # Highlight checkpoint 5
            highlight = Circle((x, y_main), 0.35, facecolor='none',
                             edgecolor='#e74c3c', linewidth=3, alpha=0.9, zorder=1)
            ax.add_patch(highlight)

    ax.text(7, 7, 'Main Timeline (Original Execution)', fontsize=9, ha='center',
           weight='bold', style='italic')

    # Time-travel arrow
    x_cp5 = 0.5 + 5 * 1.2
    x_cp10 = 0.5 + 10 * 1.2

    ax.annotate('', xy=(x_cp5, y_main + 0.3), xytext=(x_cp10, y_main + 1.5),
               arrowprops=dict(arrowstyle='->', lw=3, color='#e74c3c',
                             connectionstyle="arc3,rad=.4"))
    ax.text(9, 7.3, 'Time-Travel\nBack to CP₅', fontsize=8, ha='center',
           weight='bold', color='#e74c3c',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3cd', alpha=0.9))

    # Branch timeline
    y_branch = 4
    branch_checkpoints = [(5 + i, f'CP{5+i}\'', '#27ae60') for i in range(6)]

    ax.plot([x_cp5, 13.5], [y_branch, y_branch], 'g--', linewidth=2, alpha=0.5, zorder=0)

    for i, label, color in branch_checkpoints:
        x = 0.5 + i * 1.2
        circle = Circle((x, y_branch), 0.25, facecolor=color,
                       edgecolor='black', linewidth=1.5, alpha=0.8, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y_branch-0.6, label, fontsize=7, ha='center', weight='bold')

    ax.text(9, 3.2, 'Alternative Branch (New Execution)', fontsize=9, ha='center',
           weight='bold', style='italic', color='#27ae60')

    # Branch creation arrow
    ax.annotate('', xy=(x_cp5, y_branch + 0.3), xytext=(x_cp5, y_main - 0.3),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='#27ae60'))
    ax.text(x_cp5 - 0.7, 5, 'Branch\nHere', fontsize=7, ha='center',
           weight='bold', color='#27ae60')

    # State versioning
    version_box = FancyBboxPatch((0.3, 0.3), 5, 1.8, boxstyle="round,pad=0.1",
                                 facecolor='#e8f4f8', edgecolor='#2E86AB', linewidth=2)
    ax.add_patch(version_box)
    ax.text(2.8, 2, 'State Versioning', fontsize=10, ha='center', weight='bold')
    ax.text(2.8, 1.65, 'CP₅: {messages: [...], step: 5}', fontsize=7, ha='center', family='monospace')
    ax.text(2.8, 1.35, 'CP₁₀: {messages: [...], step: 10}', fontsize=7, ha='center', family='monospace')
    ax.text(2.8, 1.05, 'CP₅\': {messages: [...], step: 5}', fontsize=7, ha='center', family='monospace')
    ax.text(2.8, 0.65, 'All versions persisted', fontsize=7, ha='center', style='italic')

    # Use cases
    usecase_box = FancyBboxPatch((5.8, 0.3), 7.9, 1.8, boxstyle="round,pad=0.1",
                                 facecolor='#d4edda', edgecolor='#28a745', linewidth=2)
    ax.add_patch(usecase_box)
    ax.text(9.75, 2, 'Use Cases', fontsize=10, ha='center', weight='bold')

    cases = [
        '🐛 Debug: Go back to error point',
        '🔄 Replay: Test different decisions',
        '🌳 Explore: Try multiple paths from checkpoint'
    ]

    for i, case in enumerate(cases):
        ax.text(9.75, 1.6 - i*0.4, case, fontsize=7, ha='center')

    plt.tight_layout()
    plt.savefig(output_dir / 'checkpoint_timeline.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: checkpoint_timeline.png")

def main():
    """Generate all LangGraph visualizations"""
    print("=" * 60)
    print("Generating LangGraph Visualizations")
    print("=" * 60)

    create_state_management()
    create_graph_execution()
    create_conditional_routing()
    create_multi_agent()
    create_human_in_loop()
    create_react_loop()
    create_plan_execute()
    create_checkpoint_timeline()

    print("=" * 60)
    print(f"✓ All visualizations saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
