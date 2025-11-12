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

def create_vector_database_architecture():
    """
    Visualization 3: Vector Database Architecture
    Shows: Indexing structure and ANN search algorithms
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Section 12: Vector Database Architecture', fontsize=16, fontweight='bold', y=1.02)
    
    # Panel 1: Vector DB Components
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Vector Database Components', fontsize=13, fontweight='bold', pad=20)
    
    # Component 1: Storage Layer (bottom)
    storage_color = '#95a5a6'
    ax1.add_patch(plt.Rectangle((0.5, 0.5), 9, 1.5, 
                                 facecolor=storage_color, edgecolor='black', linewidth=2, alpha=0.7))
    ax1.text(5, 1.25, 'Storage Layer: Document Embeddings', fontsize=11, fontweight='bold', 
             ha='center', va='center')
    ax1.text(5, 0.8, r'$\{(\text{doc}_i, \mathbf{e}_i, \text{metadata}_i)\}_{i=1}^N$', fontsize=10, 
             ha='center', va='center')
    
    # Component 2: Index Structure (middle)
    index_color = '#3498db'
    ax1.add_patch(plt.Rectangle((0.5, 2.5), 4, 2.5, 
                                 facecolor=index_color, edgecolor='black', linewidth=2, alpha=0.7))
    ax1.text(2.5, 4, 'HNSW Index', fontsize=11, fontweight='bold', 
             ha='center', va='center', color='white')
    ax1.text(2.5, 3.5, 'Hierarchical\nNavigable\nSmall Worlds', fontsize=9, 
             ha='center', va='center', color='white')
    ax1.text(2.5, 2.8, 'O(log N) search', fontsize=9, 
             ha='center', va='center', color='white', style='italic')
    
    ax1.add_patch(plt.Rectangle((5.5, 2.5), 4, 2.5, 
                                 facecolor='#e74c3c', edgecolor='black', linewidth=2, alpha=0.7))
    ax1.text(7.5, 4, 'IVF Index', fontsize=11, fontweight='bold', 
             ha='center', va='center', color='white')
    ax1.text(7.5, 3.5, 'Inverted File\nClustering', fontsize=9, 
             ha='center', va='center', color='white')
    ax1.text(7.5, 2.8, 'Fast, less accurate', fontsize=9, 
             ha='center', va='center', color='white', style='italic')
    
    # Component 3: Query Processing (top)
    query_color = '#2ecc71'
    ax1.add_patch(plt.Rectangle((0.5, 5.5), 9, 1.5, 
                                 facecolor=query_color, edgecolor='black', linewidth=2, alpha=0.7))
    ax1.text(5, 6.5, 'Query Processing: ANN Search', fontsize=11, fontweight='bold', 
             ha='center', va='center', color='white')
    ax1.text(5, 6, r'Input: $\mathbf{q}$ → Output: top-k nearest neighbors', fontsize=10, 
             ha='center', va='center', color='white')
    
    # Component 4: Metadata Filtering (right side)
    filter_color = '#f39c12'
    ax1.add_patch(plt.Rectangle((0.5, 7.5), 9, 1.8, 
                                 facecolor=filter_color, edgecolor='black', linewidth=2, alpha=0.7))
    ax1.text(5, 8.7, 'Metadata Filtering', fontsize=11, fontweight='bold', 
             ha='center', va='center', color='white')
    ax1.text(5, 8.2, 'Filter by: date, author, category, language', fontsize=9, 
             ha='center', va='center', color='white')
    ax1.text(5, 7.8, 'Pre-filter → ANN search (hybrid approach)', fontsize=9, 
             ha='center', va='center', color='white')
    
    # Arrows showing data flow
    ax1.annotate('', xy=(5, 2.3), xytext=(5, 2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax1.annotate('', xy=(5, 5.3), xytext=(5, 5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax1.annotate('', xy=(5, 7.3), xytext=(5, 7),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Panel 2: Search Algorithm Performance Comparison
    ax2 = axes[1]
    
    algorithms = ['Exact\nSearch\n(Brute Force)', 'HNSW\n(Graph-based)', 
                  'IVF\n(Clustering)', 'LSH\n(Hashing)']
    search_time = [1000, 15, 8, 5]  # ms for 1M vectors
    recall_at_10 = [100, 98, 92, 85]  # percentage
    memory_gb = [8, 12, 6, 4]  # GB for 1M 768-dim vectors
    
    x = np.arange(len(algorithms))
    
    fig2, ax2_twin1 = plt.subplots(figsize=(7, 6))
    ax2_twin2 = ax2_twin1.twinx()
    
    # Bar chart for search time
    bars1 = ax2_twin1.bar(x - 0.2, search_time, 0.4, label='Search Time (ms)', 
                          color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Line chart for recall
    line1 = ax2_twin2.plot(x, recall_at_10, 'o-', color='#2ecc71', linewidth=3, 
                           markersize=10, label='Recall@10 (%)', markeredgecolor='black', markeredgewidth=1.5)
    
    ax2_twin1.set_ylabel('Search Time (ms, log scale)', fontsize=11, fontweight='bold', color='#e74c3c')
    ax2_twin1.set_yscale('log')
    ax2_twin1.tick_params(axis='y', labelcolor='#e74c3c')
    ax2_twin1.set_ylim(1, 2000)
    
    ax2_twin2.set_ylabel('Recall@10 (%)', fontsize=11, fontweight='bold', color='#2ecc71')
    ax2_twin2.tick_params(axis='y', labelcolor='#2ecc71')
    ax2_twin2.set_ylim(80, 105)
    
    ax2_twin1.set_xticks(x)
    ax2_twin1.set_xticklabels(algorithms, fontsize=10, fontweight='bold')
    ax2_twin1.set_xlabel('ANN Algorithm', fontsize=11, fontweight='bold')
    
    # Add value labels
    for i, (bar, recall) in enumerate(zip(bars1, recall_at_10)):
        height = bar.get_height()
        ax2_twin1.text(bar.get_x() + bar.get_width()/2., height * 1.3,
                      f'{int(height)}ms',
                      ha='center', va='bottom', fontsize=9, fontweight='bold', color='#e74c3c')
        ax2_twin2.text(i, recall + 1.5, f'{recall}%',
                      ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2ecc71')
    
    # Add title and grid
    ax2_twin1.set_title('ANN Algorithm Trade-offs\n(1M vectors, 768-dim embeddings)', 
                        fontsize=12, fontweight='bold', pad=15)
    ax2_twin1.grid(True, alpha=0.3, axis='y')
    
    # Add combined legend
    lines1, labels1 = ax2_twin1.get_legend_handles_labels()
    lines2, labels2 = ax2_twin2.get_legend_handles_labels()
    ax2_twin1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10, framealpha=0.9)
    
    # Close the twin axis subplot and use original axes[1]
    plt.close(fig2)
    
    # Recreate in original panel
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    bars1 = ax2.bar(x - 0.2, search_time, 0.4, label='Search Time (ms)', 
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    line1 = ax2_twin.plot(x, recall_at_10, 'o-', color='#2ecc71', linewidth=3, 
                          markersize=10, label='Recall@10 (%)', markeredgecolor='black', markeredgewidth=1.5)
    
    ax2.set_ylabel('Search Time (ms, log scale)', fontsize=11, fontweight='bold', color='#e74c3c')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.set_ylim(1, 2000)
    
    ax2_twin.set_ylabel('Recall@10 (%)', fontsize=11, fontweight='bold', color='#2ecc71')
    ax2_twin.tick_params(axis='y', labelcolor='#2ecc71')
    ax2_twin.set_ylim(80, 105)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, fontsize=10, fontweight='bold')
    ax2.set_xlabel('ANN Algorithm', fontsize=11, fontweight='bold')
    ax2.set_title('ANN Algorithm Trade-offs\n(1M vectors, 768-dim embeddings)', 
                  fontsize=12, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, recall) in enumerate(zip(bars1, recall_at_10)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.3,
                f'{int(height)}ms',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#e74c3c')
        ax2_twin.text(i, recall + 1.5, f'{recall}%',
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2ecc71')
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/vector_database_architecture.png", dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/vector_database_architecture.png")
    plt.close()


def create_retrieval_strategies():
    """
    Visualization 4: RAG Retrieval Strategies
    Shows: Dense vs Sparse vs Hybrid retrieval
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Section 12: Retrieval Strategies Comparison', fontsize=16, fontweight='bold', y=1.02)
    
    # Panel 1: Retrieval Methods Comparison
    ax1 = axes[0]
    
    methods = ['Dense\nRetrieval\n(Embeddings)', 'Sparse\nRetrieval\n(BM25)', 
               'Hybrid\nRetrieval\n(Both)']
    semantic_understanding = [95, 45, 92]  # Percentage
    keyword_matching = [60, 98, 95]  # Percentage
    speed_ms = [15, 3, 18]  # ms per query
    
    x = np.arange(len(methods))
    width = 0.25
    
    bars1 = ax1.bar(x - width, semantic_understanding, width, label='Semantic Understanding', 
                    color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x, keyword_matching, width, label='Keyword Matching', 
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = ax1.bar(x + width, [100-s/5 for s in speed_ms], width, label='Speed Score', 
                    color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Effectiveness Score (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Retrieval Method Trade-offs', fontsize=13, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=10, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 110)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Add recommendation box
    ax1.text(1, 105, 'Recommended:\nHybrid Retrieval', 
             ha='center', va='top', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    
    # Panel 2: Top-k Retrieval Impact on Performance
    ax2 = axes[1]
    
    k_values = np.array([1, 3, 5, 10, 20, 50])
    recall = np.array([42, 68, 78, 88, 94, 97])  # Recall percentage
    latency = k_values * 0.5 + 5  # ms (linear with k)
    relevance_score = 100 - (k_values - 1) * 1.2  # Decreases with more docs
    
    ax2_twin = ax2.twinx()
    
    # Line plots
    line1 = ax2.plot(k_values, recall, 'o-', color='#2ecc71', linewidth=3, 
                     markersize=10, label='Recall@k (%)', markeredgecolor='black', markeredgewidth=1.5)
    line2 = ax2.plot(k_values, relevance_score, 's-', color='#3498db', linewidth=3, 
                     markersize=8, label='Avg Relevance Score', markeredgecolor='black', markeredgewidth=1.5)
    line3 = ax2_twin.plot(k_values, latency, '^--', color='#e74c3c', linewidth=3, 
                          markersize=8, label='Latency (ms)', markeredgecolor='black', markeredgewidth=1.5)
    
    ax2.set_xlabel('Number of Retrieved Documents (k)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Recall / Relevance Score (%)', fontsize=11, fontweight='bold', color='#2ecc71')
    ax2.tick_params(axis='y', labelcolor='#2ecc71')
    ax2.set_ylim(0, 105)
    
    ax2_twin.set_ylabel('Latency (ms)', fontsize=11, fontweight='bold', color='#e74c3c')
    ax2_twin.tick_params(axis='y', labelcolor='#e74c3c')
    ax2_twin.set_ylim(0, 35)
    
    ax2.set_title('Top-k Selection Trade-off\n(Quality vs Speed)', 
                  fontsize=12, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values)
    
    # Add optimal k indicator
    ax2.axvline(x=10, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax2.text(10, 50, 'Optimal k=10\n(88% recall)', 
             ha='center', va='center', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/retrieval_strategies.png", dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/retrieval_strategies.png")
    plt.close()


if __name__ == "__main__":
    print(f"Generating Section 12 visualizations: Vector Databases & Retrieval...")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    create_vector_database_architecture()
    create_retrieval_strategies()
    
    print(f"\n✓ All vector database visualizations generated successfully!")
    print(f"✓ Total images: 2")
    print(f"✓ Output directory: {OUTPUT_DIR}")
