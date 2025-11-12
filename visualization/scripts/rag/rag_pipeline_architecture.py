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

def create_rag_pipeline_overview():
    """
    Visualization 1: RAG Three-Stage Pipeline
    Shows: Indexing → Retrieval → Generation flow
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Section 12: RAG Pipeline Architecture', fontsize=16, fontweight='bold', y=1.02)

    # Panel 1: Three-Stage Pipeline Flow
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('RAG Three-Stage Pipeline', fontsize=13, fontweight='bold', pad=20)

    # Stage 1: Indexing (top)
    indexing_color = '#3498db'
    ax1.add_patch(plt.Rectangle((0.5, 7), 9, 2,
                                 facecolor=indexing_color, edgecolor='black', linewidth=2, alpha=0.7))
    ax1.text(5, 8.5, 'Stage 1: Indexing', fontsize=12, fontweight='bold',
             ha='center', va='center', color='white')
    ax1.text(5, 8, 'Documents → Embeddings → Vector DB', fontsize=10,
             ha='center', va='center', color='white')
    ax1.text(5, 7.4, r'$\mathcal{D} = \{(d_i, \mathbf{e}_i)\}_{i=1}^N$', fontsize=11,
             ha='center', va='center', color='white')

    # Arrow 1→2
    ax1.annotate('', xy=(5, 6.8), xytext=(5, 7),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    # Stage 2: Retrieval (middle)
    retrieval_color = '#e74c3c'
    ax1.add_patch(plt.Rectangle((0.5, 4), 9, 2,
                                 facecolor=retrieval_color, edgecolor='black', linewidth=2, alpha=0.7))
    ax1.text(5, 5.5, 'Stage 2: Retrieval', fontsize=12, fontweight='bold',
             ha='center', va='center', color='white')
    ax1.text(5, 5, 'Query → Top-k Similar Documents', fontsize=10,
             ha='center', va='center', color='white')
    ax1.text(5, 4.4, r'$\text{top-}k(\text{sim}(q, d_i))$', fontsize=11,
             ha='center', va='center', color='white')

    # Arrow 2→3
    ax1.annotate('', xy=(5, 3.8), xytext=(5, 4),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    # Stage 3: Generation (bottom)
    generation_color = '#2ecc71'
    ax1.add_patch(plt.Rectangle((0.5, 1), 9, 2,
                                 facecolor=generation_color, edgecolor='black', linewidth=2, alpha=0.7))
    ax1.text(5, 2.5, 'Stage 3: Generation', fontsize=12, fontweight='bold',
             ha='center', va='center', color='white')
    ax1.text(5, 2, 'LLM(Context + Query) → Response', fontsize=10,
             ha='center', va='center', color='white')
    ax1.text(5, 1.4, r'$P(y | q, \text{context})$', fontsize=11,
             ha='center', va='center', color='white')

    # Output indicator
    ax1.annotate('', xy=(5, 0.8), xytext=(5, 1),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax1.text(5, 0.3, '✓ Factual Response', fontsize=11, fontweight='bold',
             ha='center', va='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # Panel 2: RAG vs Standard LLM Comparison
    ax2 = axes[1]

    methods = ['Standard\nLLM', 'RAG\nPipeline']
    hallucination_rate = [35, 8]  # Percentage
    knowledge_cutoff = [100, 15]  # Limited vs current (inverse scale)
    domain_accuracy = [60, 92]    # Percentage

    x = np.arange(len(methods))
    width = 0.25

    bars1 = ax2.bar(x - width, hallucination_rate, width, label='Hallucination Rate (%)',
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x, knowledge_cutoff, width, label='Knowledge Cutoff (months old)',
                    color='#f39c12', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = ax2.bar(x + width, domain_accuracy, width, label='Domain Accuracy (%)',
                    color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax2.set_ylabel('Score / Percentage', fontsize=11, fontweight='bold')
    ax2.set_title('RAG Benefits: Problem Mitigation', fontsize=13, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 120)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add annotation
    ax2.text(0.5, 110, '↓ 77% Reduction\nin Hallucinations',
             ha='center', va='center', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rag_pipeline_architecture.png", dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/rag_pipeline_architecture.png")
    plt.close()


def create_embedding_similarity_metrics():
    """
    Visualization 2: Embedding Similarity Metrics Comparison
    Shows: Cosine vs Euclidean vs Dot Product behavior
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Section 12: Embedding Similarity Metrics', fontsize=16, fontweight='bold', y=1.02)

    # Panel 1: Similarity Metric Comparison with Document Length
    ax1 = axes[0]

    doc_lengths = np.linspace(0.5, 3, 100)  # Document embedding magnitudes
    angle_rad = np.pi / 6  # 30 degrees, fixed semantic similarity

    # Cosine similarity (constant - scale invariant)
    cosine_sim = np.ones_like(doc_lengths) * np.cos(angle_rad)

    # Dot product (linear growth with length)
    dot_product = doc_lengths * np.cos(angle_rad)

    # Euclidean distance converted to similarity: 1 / (1 + distance)
    # distance = sqrt(1 + doc_length^2 - 2*doc_length*cos(angle))
    euclidean_dist = np.sqrt(1 + doc_lengths**2 - 2*doc_lengths*np.cos(angle_rad))
    euclidean_sim = 1 / (1 + euclidean_dist)

    ax1.plot(doc_lengths, cosine_sim, 'b-', linewidth=3, label='Cosine Similarity (Scale-Invariant)', marker='o', markersize=4, markevery=10)
    ax1.plot(doc_lengths, dot_product / 3, 'r--', linewidth=3, label='Dot Product (Favors Long Docs)', marker='s', markersize=4, markevery=10)
    ax1.plot(doc_lengths, euclidean_sim, 'g-.', linewidth=3, label='Euclidean Similarity', marker='^', markersize=4, markevery=10)

    ax1.set_xlabel('Document Embedding Magnitude (Length)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Similarity Score', fontsize=11, fontweight='bold')
    ax1.set_title('How Metrics Respond to Document Length\n(Same Semantic Angle θ=30°)',
                  fontsize=12, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.2)

    # Add annotations
    ax1.axhline(y=np.cos(angle_rad), color='blue', linestyle=':', alpha=0.5, linewidth=2)
    ax1.text(1.5, 0.92, 'Cosine: Constant\n(Ideal for RAG)',
             fontsize=9, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    ax1.axvline(x=1.0, color='gray', linestyle=':', alpha=0.3)
    ax1.text(1.0, 0.1, 'Query\nLength', fontsize=8, ha='center', va='bottom')

    # Panel 2: Concrete Example - Similarity Scores
    ax2 = axes[1]

    # Example: Query vs 3 documents
    examples = ['Query vs\nDoc1\n(Relevant)', 'Query vs\nDoc2\n(Related)', 'Query vs\nDoc3\n(Unrelated)']
    cosine_scores = [0.982, 0.658, 0.134]
    dot_scores = [0.875, 0.523, 0.089]
    euclidean_scores = [0.912, 0.601, 0.187]

    x = np.arange(len(examples))
    width = 0.25

    bars1 = ax2.bar(x - width, cosine_scores, width, label='Cosine Similarity',
                    color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x, dot_scores, width, label='Dot Product',
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = ax2.bar(x + width, euclidean_scores, width, label='Euclidean (Normalized)',
                    color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax2.set_ylabel('Similarity Score', fontsize=11, fontweight='bold')
    ax2.set_title('Worked Example: 3D Embedding Space\nQuery = [0.5, 0.3, 0.2]',
                  fontsize=12, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(examples, fontsize=9, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.1)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Add retrieval decision
    ax2.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, linewidth=2)
    ax2.text(2.5, 0.72, 'Retrieval Threshold\n(top-k cutoff)',
             fontsize=8, ha='center', va='bottom',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/embedding_similarity_metrics.png", dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/embedding_similarity_metrics.png")
    plt.close()


if __name__ == "__main__":
    print(f"Generating Section 12 visualizations: RAG Pipeline & Architecture...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    create_rag_pipeline_overview()
    create_embedding_similarity_metrics()

    print(f"\n✓ All RAG pipeline visualizations generated successfully!")
    print(f"✓ Total images: 2")
    print(f"✓ Output directory: {OUTPUT_DIR}")
