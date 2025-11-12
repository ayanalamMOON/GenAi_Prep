import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

plt.style.use('seaborn-v0_8-darkgrid')

# Output directory (two levels up from scripts/rag/)
OUTPUT_DIR = "../../images/rag"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# High-resolution publication quality
DPI = 300

def create_advanced_rag_methods():
    """
    Visualization 5: Advanced RAG Architectures
    Shows: HyDE, Self-RAG, RAPTOR comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Section 12: Advanced RAG Architectures', fontsize=16, fontweight='bold', y=1.02)

    # Panel 1: Architecture Comparison
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Evolution of RAG Methods', fontsize=13, fontweight='bold', pad=20)

    # Method 1: Naive RAG (bottom)
    naive_color = '#95a5a6'
    ax1.add_patch(plt.Rectangle((0.5, 0.5), 9, 1.8,
                                 facecolor=naive_color, edgecolor='black', linewidth=2, alpha=0.7))
    ax1.text(5, 1.7, 'Naive RAG', fontsize=11, fontweight='bold', ha='center', va='center')
    ax1.text(5, 1.2, 'Query → Retrieve → Generate', fontsize=9, ha='center', va='center')
    ax1.text(5, 0.8, 'Simple, fast, but limited quality', fontsize=8, ha='center', va='center', style='italic')

    # Method 2: HyDE (Hypothetical Document Embeddings)
    hyde_color = '#3498db'
    ax1.add_patch(plt.Rectangle((0.5, 2.8), 9, 1.8,
                                 facecolor=hyde_color, edgecolor='black', linewidth=2, alpha=0.7))
    ax1.text(5, 4, 'HyDE (Hypothetical Docs)', fontsize=11, fontweight='bold',
             ha='center', va='center', color='white')
    ax1.text(5, 3.5, 'Query → Generate Hypothetical Answer', fontsize=9,
             ha='center', va='center', color='white')
    ax1.text(5, 3.1, '→ Embed Answer → Retrieve Similar', fontsize=9,
             ha='center', va='center', color='white')

    # Method 3: Self-RAG
    selfrag_color = '#e74c3c'
    ax1.add_patch(plt.Rectangle((0.5, 5.1), 9, 1.8,
                                 facecolor=selfrag_color, edgecolor='black', linewidth=2, alpha=0.7))
    ax1.text(5, 6.3, 'Self-RAG (Self-Reflective)', fontsize=11, fontweight='bold',
             ha='center', va='center', color='white')
    ax1.text(5, 5.8, 'Retrieve → Reflect (is this relevant?)', fontsize=9,
             ha='center', va='center', color='white')
    ax1.text(5, 5.4, '→ Generate → Critique → Refine', fontsize=9,
             ha='center', va='center', color='white')

    # Method 4: RAPTOR (top)
    raptor_color = '#2ecc71'
    ax1.add_patch(plt.Rectangle((0.5, 7.4), 9, 1.8,
                                 facecolor=raptor_color, edgecolor='black', linewidth=2, alpha=0.7))
    ax1.text(5, 8.6, 'RAPTOR (Recursive Abstraction)', fontsize=11, fontweight='bold',
             ha='center', va='center', color='white')
    ax1.text(5, 8.1, 'Build hierarchical summary tree', fontsize=9,
             ha='center', va='center', color='white')
    ax1.text(5, 7.7, '→ Retrieve from multiple abstraction levels', fontsize=9,
             ha='center', va='center', color='white')

    # Arrows showing evolution
    for y in [2.3, 4.6, 6.9]:
        ax1.annotate('', xy=(0.3, y + 0.5), xytext=(0.3, y),
                    arrowprops=dict(arrowstyle='->', lw=3, color='black'))
        ax1.text(0.15, y + 0.25, 'Better', fontsize=8, ha='center', va='center',
                rotation=90, fontweight='bold')

    # Panel 2: Performance Comparison
    ax2 = axes[1]

    methods = ['Naive\nRAG', 'HyDE', 'Self-RAG', 'RAPTOR']
    answer_quality = [72, 81, 88, 92]  # Percentage
    latency_ms = [120, 280, 450, 380]  # ms per query
    complexity = [1, 2.5, 4, 3.5]  # Relative complexity (1-5 scale)

    x = np.arange(len(methods))

    # Create twin axis
    ax2_twin = ax2.twinx()

    # Bar chart for answer quality
    bars1 = ax2.bar(x - 0.2, answer_quality, 0.4, label='Answer Quality (%)',
                    color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)

    # Line chart for latency
    line1 = ax2_twin.plot(x, latency_ms, 'o-', color='#e74c3c', linewidth=3,
                          markersize=10, label='Latency (ms)', markeredgecolor='black', markeredgewidth=1.5)

    ax2.set_ylabel('Answer Quality (%)', fontsize=11, fontweight='bold', color='#2ecc71')
    ax2.tick_params(axis='y', labelcolor='#2ecc71')
    ax2.set_ylim(60, 100)

    ax2_twin.set_ylabel('Latency (ms)', fontsize=11, fontweight='bold', color='#e74c3c')
    ax2_twin.tick_params(axis='y', labelcolor='#e74c3c')
    ax2_twin.set_ylim(0, 500)

    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=10, fontweight='bold')
    ax2.set_xlabel('RAG Architecture', fontsize=11, fontweight='bold')
    ax2.set_title('Quality vs Latency Trade-off', fontsize=12, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2ecc71')

    for i, lat in enumerate(latency_ms):
        ax2_twin.text(i, lat + 20, f'{lat}ms',
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='#e74c3c')

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10, framealpha=0.9)

    # Add recommendation
    ax2.text(2.5, 65, 'Self-RAG: Best\nQuality/Latency',
             ha='center', va='center', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/advanced_rag_architectures.png", dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/advanced_rag_architectures.png")
    plt.close()


def create_reranking_impact():
    """
    Visualization 6: Reranking Model Impact
    Shows: Retrieval + Reranking performance improvement
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Section 12: Reranking Model Impact on RAG Quality', fontsize=16, fontweight='bold', y=1.02)

    # Panel 1: Reranking Pipeline
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Two-Stage Retrieval Pipeline', fontsize=13, fontweight='bold', pad=20)

    # Stage 1: Initial Retrieval (top)
    retrieval_color = '#3498db'
    ax1.add_patch(plt.Rectangle((0.5, 6.5), 9, 2.5,
                                 facecolor=retrieval_color, edgecolor='black', linewidth=2, alpha=0.7))
    ax1.text(5, 8.3, 'Stage 1: Initial Retrieval (Fast)', fontsize=11, fontweight='bold',
             ha='center', va='center', color='white')
    ax1.text(5, 7.7, 'ANN Search → Retrieve top-100 candidates', fontsize=10,
             ha='center', va='center', color='white')
    ax1.text(5, 7.2, 'Speed: ~10ms | Recall@100: 95%', fontsize=9,
             ha='center', va='center', color='white')

    # Arrow
    ax1.annotate('', xy=(5, 6.3), xytext=(5, 6.5),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax1.text(6, 6.4, 'Filter: 100 → 10', fontsize=9, ha='left', va='center', fontweight='bold')

    # Stage 2: Reranking (middle)
    rerank_color = '#e74c3c'
    ax1.add_patch(plt.Rectangle((0.5, 3.5), 9, 2.5,
                                 facecolor=rerank_color, edgecolor='black', linewidth=2, alpha=0.7))
    ax1.text(5, 5.3, 'Stage 2: Reranking (Accurate)', fontsize=11, fontweight='bold',
             ha='center', va='center', color='white')
    ax1.text(5, 4.7, 'Cross-Encoder → Score each pair (query, doc)', fontsize=10,
             ha='center', va='center', color='white')
    ax1.text(5, 4.2, 'Speed: +50ms | Precision@10: 92%', fontsize=9,
             ha='center', va='center', color='white')

    # Arrow
    ax1.annotate('', xy=(5, 3.3), xytext=(5, 3.5),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax1.text(6, 3.4, 'Top-10 ranked', fontsize=9, ha='left', va='center', fontweight='bold')

    # Stage 3: Generation (bottom)
    generation_color = '#2ecc71'
    ax1.add_patch(plt.Rectangle((0.5, 0.5), 9, 2.5,
                                 facecolor=generation_color, edgecolor='black', linewidth=2, alpha=0.7))
    ax1.text(5, 2.3, 'Stage 3: Context-Aware Generation', fontsize=11, fontweight='bold',
             ha='center', va='center', color='white')
    ax1.text(5, 1.7, 'LLM generates answer from top-10 reranked docs', fontsize=10,
             ha='center', va='center', color='white')
    ax1.text(5, 1.2, 'Final Answer Quality: +18% vs no reranking', fontsize=9,
             ha='center', va='center', color='white')

    # Total time annotation
    ax1.text(9.5, 5, 'Total:\n~60ms', fontsize=10, ha='center', va='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

    # Panel 2: Reranking Impact on Metrics
    ax2 = axes[1]

    metrics = ['Precision@10', 'NDCG@10', 'MRR', 'Answer\nQuality']
    without_reranking = [65, 0.68, 0.62, 74]
    with_reranking = [92, 0.89, 0.84, 92]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax2.bar(x - width/2, without_reranking, width, label='Without Reranking',
                    color='#95a5a6', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, with_reranking, width, label='With Reranking',
                    color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax2.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Reranking Performance Gains', fontsize=12, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=10, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 105)

    # Add value labels and improvement percentages
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()

        ax2.text(bar1.get_x() + bar1.get_width()/2., height1,
                f'{int(height1)}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax2.text(bar2.get_x() + bar2.get_width()/2., height2,
                f'{int(height2)}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Improvement percentage
        improvement = ((height2 - height1) / height1) * 100
        ax2.text(i, max(height1, height2) + 3, f'+{improvement:.0f}%',
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='#2ecc71')

    # Add average improvement annotation
    avg_improvement = np.mean([((with_reranking[i] - without_reranking[i]) / without_reranking[i]) * 100
                               for i in range(len(metrics))])
    ax2.text(1.5, 15, f'Avg Improvement:\n+{avg_improvement:.0f}%',
             ha='center', va='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/reranking_impact.png", dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/reranking_impact.png")
    plt.close()


if __name__ == "__main__":
    print(f"Generating Section 12 visualizations: Advanced RAG & Reranking...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    create_advanced_rag_methods()
    create_reranking_impact()

    print(f"\n✓ All advanced RAG visualizations generated successfully!")
    print(f"✓ Total images: 2")
    print(f"✓ Output directory: {OUTPUT_DIR}")
