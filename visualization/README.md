# LLM Study Material Visualizations

This directory contains Python scripts to generate visualizations for the LLM Study Material LaTeX document.

## Directory Structure

```
visualization/
├── scripts/                                      # Python visualization scripts (organized by topic)
│   ├── evaluation/                               # Section 10: Evaluation & Benchmarking (8 scripts)
│   │   ├── evaluation_benchmarks.py
│   │   ├── evaluation_best_practices.py
│   │   ├── evaluation_bleu_rouge.py
│   │   ├── evaluation_error_analysis.py
│   │   ├── evaluation_human_eval.py
│   │   ├── evaluation_multitask.py
│   │   ├── evaluation_perplexity.py
│   │   └── evaluation_visualizations.py
│   ├── fine_tuning/                              # Section 1: Fine-Tuning (1 script, 10 visualizations)
│   │   └── fine_tuning_visualizations.py
│   ├── instruction_finetuning/                   # Section 9: Instruction Fine-Tuning (1 script, 6 visualizations)
│   │   └── instruction_finetuning_visualizations.py
│   ├── langchain/                                # LangChain Section (1 script, 6 visualizations)
│   │   └── langchain_visualizations.py
│   ├── langgraph/                                # LangGraph Section (1 script, 8 visualizations)
│   │   └── langgraph_visualizations.py
│   ├── lora/                                     # LoRA Section (1 script, 9 visualizations)
│   │   └── lora_visualizations.py
│   ├── pretraining/                              # Pre-training Section (1 script, 8 visualizations)
│   │   └── pretraining_visualizations.py
│   ├── qlora/                                    # QLoRA Section (1 script, 6 visualizations)
│   │   └── qlora_visualizations.py
│   ├── rag/                                      # Section 12: RAG Systems (3 scripts, 6 visualizations)
│   │   ├── rag_pipeline_architecture.py
│   │   ├── rag_vector_databases.py
│   │   └── rag_advanced_architectures.py
│   └── safety/                                   # Section 11: Safety, Ethics & Bias Mitigation (6 scripts)
│       ├── safety_bias_detection.py
│       ├── safety_fairness_constraints.py
│       ├── safety_fairness_metrics.py
│       ├── safety_mitigation_strategies.py
│       ├── safety_monitoring_dashboard.py
│       └── safety_toxicity_detection.py
├── images/                                       # Generated visualization images (matches script structure)
│   ├── evaluation/                               # Section 10 visualizations (8 PNG files, 4.6 MB)
│   ├── fine_tuning/                              # Section 1 visualizations (10 PNG files, 2.9 MB)
│   ├── instruction_finetuning/                   # Section 9 visualizations (6 PNG files, 2.9 MB)
│   ├── langchain/                                # LangChain visualizations (6 PNG files)
│   ├── langgraph/                                # LangGraph visualizations (8 PNG files)
│   ├── lora/                                     # LoRA visualizations (9 PNG files)
│   ├── pretraining/                              # Pre-training visualizations (8 PNG files)
│   ├── qlora/                                    # QLoRA visualizations (6 PNG files)
│   ├── rag/                                      # Section 12 visualizations (6 PNG files, 2.1 MB)
│   └── safety/                                   # Section 11 visualizations (6 PNG files, 2.8 MB)
└── README.md                                      # This file
```

**Note:** Scripts are now organized into topic-wise folders for better maintainability. Each script folder corresponds to a section in the LaTeX document.

## Setup

### 1. Create Virtual Environment

```bash
cd GenAi_Prep
python -m venv venv
source venv/Scripts/activate  # On Windows
# or
source venv/bin/activate      # On Linux/Mac
```

### 2. Install Dependencies

```bash
pip install matplotlib seaborn numpy
```

## Generating Visualizations

### General Pattern

All visualization scripts follow the same execution pattern:

```bash
cd visualization/scripts/<topic_folder>
python <script_name>.py
```

### Fine-Tuning Visualizations (Section 1)

```bash
cd visualization/scripts/fine_tuning
python fine_tuning_visualizations.py
```

This generates 10 high-quality visualizations:

1. **transfer_learning_process.png** - Shows the complete pipeline from pre-training → fine-tuning → adapted model
2. **learning_rate_comparison.png** - Compares LR schedules between pre-training (3e-4) and fine-tuning (3e-5)
3. **discriminative_learning_rates.png** - Layer-wise learning rates showing earlier layers get smaller updates
4. **gradient_flow_analysis.png** - Demonstrates vanishing gradient problem with 50x difference between layers
5. **catastrophic_forgetting.png** - Side-by-side comparison of performance with/without EWC regularization
6. **fisher_information_matrix.png** - Heatmap showing parameter importance for original task
7. **training_loss_curves.png** - Different training scenarios (good convergence, overfitting, unstable)
8. **memory_comparison.png** - GPU memory requirements: Full fine-tuning vs LoRA across model sizes
9. **hyperparameter_recommendations.png** - Recommended hyperparameters by dataset size
10. **convergence_analysis.png** - Impact of learning rate on convergence speed and stability

### Instruction Fine-Tuning Visualizations (Section 9)

```bash
cd visualization/scripts/instruction_finetuning
python instruction_finetuning_visualizations.py
```

This generates 6 comprehensive visualizations:

1. **traditional_vs_instruction.png** - Side-by-side architectural comparison showing traditional fine-tuning (single task, direct input→output) vs instruction fine-tuning (multi-task with explicit instruction layer enabling 6+ task categories)
2. **dataset_formats.png** - Visual comparison of 6 popular instruction dataset formats: Alpaca (52K examples), ShareGPT (90K), FLAN (1.8M), Dolly (15K), OpenOrca (4.2M), and WizardLM (250K) with JSON structure breakdowns
3. **instruction_quality_metrics.png** - 4-panel quality analysis dashboard:
   - Radar chart comparing quality dimensions (Clarity, Specificity, Complexity, Diversity, Coherence, Relevance)
   - Dataset size vs performance curve showing logarithmic growth
   - Task category distribution in FLAN dataset (25% QA, 18% Translation, etc.)
   - Instruction length distribution histogram (mean: 70 tokens, range: 20-150)
4. **data_augmentation_pipeline.png** - Complete augmentation workflow showing 3 techniques (Paraphrasing, Complexity Variation, Format Conversion) plus Self-Instruct, Back-Translation, and Quality Filtering to multiply dataset size by 3-5x
5. **training_dynamics.png** - 4-panel training analysis:
   - Loss curves for different dataset sizes (1K, 5K, 25K, 100K)
   - Multi-task performance evolution across 10 epochs for 5 tasks
   - Learning rate schedule with warmup and cosine decay
   - Batch size impact showing optimal convergence at batch size 64
6. **evaluation_metrics_dashboard.png** - Comprehensive 9-panel evaluation dashboard:
   - Task-specific performance across 6 benchmarks (average +20% improvement)
   - Human preference win rate (65% vs 35%)
   - 3 radar charts for Knowledge, Interaction, and Safety capabilities
   - 3 distribution plots for response length, latency, and quality scores

### Evaluation & Benchmarking Visualizations (Section 10)

```bash
cd visualization/scripts/evaluation
python evaluation_benchmarks.py
python evaluation_best_practices.py
python evaluation_bleu_rouge.py
python evaluation_error_analysis.py
python evaluation_human_eval.py
python evaluation_multitask.py
python evaluation_perplexity.py
python evaluation_visualizations.py
```

This generates 8 comprehensive visualizations covering benchmarks, metrics, error analysis, and best practices.

### Safety, Ethics & Bias Mitigation Visualizations (Section 11)

```bash
cd visualization/scripts/safety
python safety_bias_detection.py
python safety_fairness_metrics.py
python safety_fairness_constraints.py
python safety_toxicity_detection.py
python safety_mitigation_strategies.py
python safety_monitoring_dashboard.py
```

This generates 6 comprehensive visualizations:

1. **bias_detection_methods.png** - WEAT effect sizes (gender/race bias) + Counterfactual perplexity analysis
2. **fairness_metrics_analysis.png** - ROC curves (equalized odds violation) + Fairness-accuracy Pareto frontier
3. **fairness_constraints_detailed.png** - Demographic parity across 8 groups + TPR/FPR breakdown
4. **toxicity_safety_analysis.png** - Toxicity score distributions + Multi-layer safety filter pipeline
5. **mitigation_strategies_comparison.png** - Debiasing methods comparison + Bias-accuracy trade-off landscape
6. **safety_monitoring_dashboard.png** - 30-day safety metrics timeline + Alert system performance

### RAG (Retrieval-Augmented Generation) Visualizations (Section 12)

```bash
cd visualization/scripts/rag
python rag_pipeline_architecture.py
python rag_vector_databases.py
python rag_advanced_architectures.py
```

This generates 6 comprehensive visualizations:

1. **rag_pipeline_architecture.png** - Three-stage RAG pipeline (Indexing → Retrieval → Generation) + RAG vs Standard LLM comparison
2. **embedding_similarity_metrics.png** - Cosine/Euclidean/Dot product behavior with document length + Worked example in 3D space
3. **vector_database_architecture.png** - Vector DB components (Storage, Index, Query) + ANN algorithm trade-offs (HNSW, IVF, LSH)
4. **retrieval_strategies.png** - Dense vs Sparse vs Hybrid retrieval comparison + Top-k selection impact on quality/speed
5. **advanced_rag_architectures.png** - Evolution of RAG methods (Naive → HyDE → Self-RAG → RAPTOR) + Quality vs Latency trade-off
6. **reranking_impact.png** - Two-stage retrieval pipeline + Reranking performance gains across metrics

### Other Topic Visualizations

**LoRA:**
```bash
cd visualization/scripts/lora
python lora_visualizations.py
```

**QLoRA:**
```bash
cd visualization/scripts/qlora
python qlora_visualizations.py
```

**Pre-training:**
```bash
cd visualization/scripts/pretraining
python pretraining_visualizations.py
```

**LangChain:**
```bash
cd visualization/scripts/langchain
python langchain_visualizations.py
```

**LangGraph:**
```bash
cd visualization/scripts/langgraph
python langgraph_visualizations.py
```

## Integration with LaTeX

All visualizations are automatically referenced in the LaTeX document at contextually appropriate locations:

- **Section 1.1.2**: Transfer learning diagram after key concepts
- **Section 1.4**: Memory comparison after method comparison table
- **Section 1.6.2**: Learning rate schedules (2 figures)
- **Section 1.6.3**: Catastrophic forgetting and Fisher Information after EWC explanation
- **Section 1.6.4**: Convergence analysis
- **Section 1.6.5**: Training loss curves after working example
- **Section 1.6.7**: Gradient flow analysis
- **Section 1.6.9**: Hyperparameter recommendations

## Customization

Each visualization script can be customized:

- **Resolution**: Change `dpi=300` parameter in `plt.savefig()` calls
- **Colors**: Modify color schemes at the top of each script
- **Size**: Adjust `figsize=(width, height)` parameters
- **Content**: Modify data values and formulas to reflect updated theories

## Output Specifications

All images are:
- **Format**: PNG with transparent background support
- **Resolution**: 300 DPI (publication quality)
- **Dimensions**: Optimized for full-width LaTeX figures (12-14 inches wide)
- **File Size**: 150-600 KB each (compressed but high quality)

## Adding New Visualizations

To add visualizations for a new topic/section:

### Step 1: Create Topic Folder Structure

```bash
cd visualization/scripts
mkdir <topic_name>  # e.g., mkdir rag or mkdir prompt_engineering
```

### Step 2: Create Visualization Script

Create a new script in the topic folder (e.g., `scripts/<topic_name>/<topic_name>_visualizations.py`):

```python
"""
Section X: <Topic Name> - Visualizations
Creates N visualizations for <topic description>
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# IMPORTANT: Output directory is ../../images/<topic_name> (two levels up from scripts/<topic_name>/)
OUTPUT_DIR = "../../images/<topic_name>"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# High-resolution publication quality
DPI = 300

def create_visualization_1():
    """
    Create first visualization with clear description
    Max 2 visualizations per PNG (use subplots(1, 2) for side-by-side)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Main Title', fontsize=16, fontweight='bold', y=1.02)

    # Panel 1: Left visualization
    ax1 = axes[0]
    # ... your plotting code ...
    ax1.set_title('(1) Panel 1 Title', fontsize=12, fontweight='bold', pad=10)

    # Panel 2: Right visualization
    ax2 = axes[1]
    # ... your plotting code ...
    ax2.set_title('(2) Panel 2 Title', fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/descriptive_name.png", dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/descriptive_name.png")
    plt.close()

if __name__ == "__main__":
    print(f"Generating Section X visualizations: <Topic Name>...")
    create_visualization_1()
    # create_visualization_2()
    # ...
    print(f"\n✓ All <topic_name> visualizations generated successfully!")
    print(f"✓ Output directory: {OUTPUT_DIR}")
```

### Step 3: Create Image Output Directory

```bash
cd visualization/images
mkdir <topic_name>  # This will be auto-created by script, but good to create manually for clarity
```

### Step 4: Generate Visualizations

```bash
cd visualization/scripts/<topic_name>
python <topic_name>_visualizations.py
```

### Step 5: Integrate into LaTeX Document

In `latex/LLM_Study_Material.tex`, add figure blocks at appropriate locations:

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{../visualization/images/<topic_name>/descriptive_name.png}
\caption{\textbf{Caption Title:} Detailed description explaining what the visualization shows, key insights, and how to interpret it. Reference specific values and thresholds shown in the image.}
\label{fig:<topic_name>_<visualization_name>}
\end{figure}
```

### Key Requirements for New Visualizations

1. **Folder Structure**: Always create `scripts/<topic_name>/` folder for scripts
2. **Output Path**: Use `OUTPUT_DIR = "../../images/<topic_name>"` (two levels up)
3. **Max 2 Panels**: Keep visualizations uncluttered - max 2 visualizations per PNG
4. **Separate Scripts**: If topic needs >3 visualizations, create multiple scripts to avoid length limits
5. **High Resolution**: Always use `DPI = 300` for publication quality
6. **Descriptive Names**: Use snake_case descriptive names (e.g., `bias_detection_methods.png`)
7. **Documentation**: Add docstrings explaining what each visualization shows
8. **Consistent Style**: Use `seaborn-v0_8-darkgrid` style for consistency

### Example: Adding RAG Visualizations

```bash
# 1. Create folders
mkdir visualization/scripts/rag
mkdir visualization/images/rag

# 2. Create script
cd visualization/scripts/rag
cat > rag_retrieval_pipeline.py << 'EOF'
"""
Section 12: RAG 2.0 - Retrieval Pipeline Visualizations
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

plt.style.use('seaborn-v0_8-darkgrid')
OUTPUT_DIR = "../../images/rag"  # Two levels up!
os.makedirs(OUTPUT_DIR, exist_ok=True)
DPI = 300

def create_retrieval_pipeline_viz():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # ... visualization code ...
    plt.savefig(f"{OUTPUT_DIR}/retrieval_pipeline.png", dpi=DPI, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_retrieval_pipeline_viz()
EOF

# 3. Generate
python rag_retrieval_pipeline.py

# 4. Add to LaTeX (in appropriate section)
```

## Best Practices

1. **Naming Convention**: Use descriptive snake_case names (e.g., `gradient_flow_analysis.png`)
2. **Documentation**: Add docstrings to each visualization function explaining what it shows
3. **Consistency**: Use similar color schemes and styles across related visualizations
4. **Context**: Place images near the relevant text/equations in LaTeX
5. **Labels**: Include clear titles, axis labels, and legends in all plots
6. **Annotations**: Add arrows and text boxes to highlight key insights

## Troubleshooting

### Import Errors
```bash
# Ensure virtual environment is activated
source venv/Scripts/activate
pip install --upgrade matplotlib seaborn numpy
```

### Image Not Showing in PDF
- Check file path is relative to LaTeX document root
- Verify image exists: `ls -la visualization/images/fine_tuning/`
- Ensure `\usepackage{graphicx}` is in LaTeX preamble

### Low Quality Images
- Increase DPI: `plt.savefig(..., dpi=600, bbox_inches='tight')`
- Use vector format: Change `.png` to `.pdf` (requires pdflatex)

## Future Enhancements

- [ ] Add visualizations for LoRA section (rank comparison, adapter diagrams)
- [ ] Add visualizations for QLoRA (quantization schemes, memory savings)
- [ ] Add visualizations for LangChain (RAG pipeline, agent workflows)
- [ ] Add visualizations for LangGraph (state graphs, workflow diagrams)
- [ ] Create interactive visualizations with Plotly
- [ ] Add animation scripts for training dynamics

## License

These visualization scripts are part of the LLM Study Material project.
All generated images are for educational purposes.
