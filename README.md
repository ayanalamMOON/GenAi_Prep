# Generative AI Study Materials

[![License: Educational](https://img.shields.io/badge/License-Educational-blue.svg)](LICENSE)
[![LaTeX](https://img.shields.io/badge/LaTeX-Study%20Guide-008080.svg)](docs/LLM_Study_Material.pdf)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg)](requirements.txt)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626.svg)](notebooks/)

> **Comprehensive university-level study guide for Large Language Models (LLMs), transformers, RLHF, fine-tuning, and generative AI techniques. Includes 335 pages of theory, 47 visualizations, and 4 practical Jupyter notebooks.**

**Keywords:** Large Language Models, LLM, Transformers, RLHF, LoRA, QLoRA, Fine-Tuning, GPT, BERT, Attention Mechanism, Pre-training, Instruction Tuning, RAG, LangChain, LangGraph, Deep Learning, Natural Language Processing, NLP, Machine Learning, AI Safety

This repository contains comprehensive study materials for understanding Large Language Models (LLMs) and generative AI techniques. The materials are designed for university-level learners and cover both theoretical foundations and practical implementations.

## Repository Structure

```
GenAi_Prep/
├── docs/                          # Documentation and compiled PDFs
│   ├── LLM_Study_Material.pdf    # Main study guide (335 pages)
│   └── GenAI-Part-2.pdf          # Supplementary materials
├── notebooks/                     # Jupyter notebooks for hands-on learning
│   ├── Finetuning_Dr_Patient.ipynb
│   ├── GPT_2_FINETUNING_QLoRA.ipynb
│   ├── LORA_FINE_TUNING.ipynb
│   └── Shoolini_Pretraining.ipynb
├── latex/                         # LaTeX source files
│   ├── LLM_Study_Material.tex
│   └── AI_Assignment_3.tex
├── visualization/                 # Educational visualizations
│   ├── images/                   # Generated diagrams (47 visualizations)
│   ├── scripts/                  # Python visualization generators
│   └── README.md
├── requirements.txt              # Python dependencies
└── README.md
```

## Primary Study Material

**`docs/LLM_Study_Material.pdf`** (335 pages)

The main study document covering end-to-end LLM development, from fundamental concepts through advanced techniques. Each section includes mathematical derivations, practical code implementations, hyperparameter guidance, and debugging strategies.

### Topics Covered

**Foundation & Architecture**
- Transformer architecture and self-attention mechanisms
- Tokenization strategies and embedding techniques
- Positional encodings and attention patterns

**Training Methodologies**
- Pre-training objectives and optimization techniques
- Fine-tuning approaches (LoRA, QLoRA, PEFT)
- Reinforcement Learning from Human Feedback (RLHF)
- Direct Preference Optimization (DPO)

**Advanced Techniques**
- Instruction fine-tuning and dataset preparation
- Model evaluation and benchmarking methodologies
- Retrieval-Augmented Generation (RAG) systems
- LangChain and LangGraph frameworks

**Production Considerations**
- Safety, ethics, and bias mitigation
- Hyperparameter tuning and optimization
- Memory efficiency and quantization
- Debugging strategies and common pitfalls

## Jupyter Notebooks

### `notebooks/Finetuning_Dr_Patient.ipynb`
Practical implementation of fine-tuning techniques using medical conversation datasets. Demonstrates data preparation, model configuration, and training loops.

### `notebooks/GPT_2_FINETUNING_QLoRA.ipynb`
Hands-on guide to parameter-efficient fine-tuning using QLoRA (Quantized Low-Rank Adaptation). Includes memory optimization strategies and performance benchmarking.

### `notebooks/LORA_FINE_TUNING.ipynb`
Implementation of Low-Rank Adaptation for efficient model fine-tuning. Covers adapter configuration, training procedures, and model merging techniques.

### `notebooks/Shoolini_Pretraining.ipynb`
Pre-training demonstration covering tokenization, data processing pipelines, and training optimization for language models from scratch.

## Visualizations

The repository includes 47 educational visualizations covering key LLM concepts:

### Fine-Tuning Workflows

<p align="center">
  <img src="./visualization/images/fine_tuning/transfer_learning_process.png" alt="Transfer Learning Process" width="700"/>
</p>

*Figure: Transfer learning process showing the transition from pre-trained models to task-specific fine-tuning.*

<p align="center">
  <img src="./visualization/images/fine_tuning/training_loss_curves.png" alt="Training Loss Curves" width="700"/>
</p>

*Figure: Training and validation loss curves demonstrating convergence behavior during fine-tuning.*

### LoRA Architecture

<p align="center">
  <img src="./visualization/images/lora/lora_architecture.png" alt="LoRA Architecture" width="700"/>
</p>

*Figure: Low-Rank Adaptation architecture showing parameter-efficient fine-tuning through low-rank matrices.*

<p align="center">
  <img src="./visualization/images/lora/rank_comparison.png" alt="Rank Comparison" width="700"/>
</p>

*Figure: Performance comparison across different rank values in LoRA, illustrating the trade-off between efficiency and model capacity.*

### LangChain Framework

<p align="center">
  <img src="./visualization/images/langchain/langchain_architecture.png" alt="LangChain Architecture" width="700"/>
</p>

*Figure: LangChain architecture demonstrating the modular components for building LLM applications.*

<p align="center">
  <img src="./visualization/images/langchain/rag_architecture.png" alt="RAG Architecture" width="700"/>
</p>

*Figure: Retrieval-Augmented Generation (RAG) architecture showing the integration of external knowledge retrieval with language generation.*

### LangGraph Execution

<p align="center">
  <img src="./visualization/images/langgraph/graph_execution.png" alt="LangGraph Execution" width="700"/>
</p>

*Figure: LangGraph execution flow demonstrating stateful, graph-based agent orchestration.*

See `visualization/README.md` for the complete collection of visualizations and usage instructions for generating custom diagrams.

## Repository Statistics

- **Study Material**: 335 pages of comprehensive content
- **Jupyter Notebooks**: 4 practical implementations
- **Visualizations**: 47 educational diagrams across 6 categories
- **Code Examples**: Production-ready implementations with extensive documentation
- **Topics Covered**: 12 major sections from fundamentals to advanced techniques

## Getting Started

### Prerequisites

- Python 3.8 or higher
- LaTeX distribution (for compiling `.tex` files from the `latex/` directory)
- Jupyter Notebook or JupyterLab
- CUDA-compatible GPU (recommended for training notebooks)

### Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd GenAi_Prep
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

**For theoretical study:**
Refer to `docs/LLM_Study_Material.pdf` for comprehensive coverage of concepts, mathematics, and implementation details.

**For practical implementation:**
Open the relevant notebook in the `notebooks/` directory and follow the step-by-step code examples. Each notebook includes detailed comments explaining the rationale behind implementation choices.

**For visualization generation:**
Navigate to the `visualization/` directory and refer to the README for script usage and customization options.

## Learning Path

For those new to LLMs, the recommended learning sequence is:

1. **Foundation** (Sections 1-3): Transformer architecture, tokenization, embeddings
2. **Pre-training** (Section 4): Language modeling objectives, optimization techniques
3. **Fine-tuning** (Sections 5-7): Transfer learning, LoRA, QLoRA, PEFT methods
4. **RLHF and Alignment** (Section 8): Reward modeling, PPO, DPO
5. **Advanced Topics** (Sections 9-12): Instruction tuning, evaluation, safety, RAG systems

Supplement theoretical study with the corresponding notebooks in `notebooks/` to reinforce understanding through implementation. Use the visualizations in `visualization/images/` as reference diagrams while studying.

## Technical Notes

### Code Quality

All code examples follow production-ready standards:
- Comprehensive inline documentation
- Explicit hyperparameter specifications with justifications
- Error handling and validation checks
- Memory optimization strategies for resource-constrained environments

### Mathematical Rigor

Mathematical derivations are presented with:
- Step-by-step progression from intuition to formalization
- Clear notation and variable definitions
- Dimensionality analysis for tensor operations
- Computational complexity considerations

## Compilation

To regenerate the PDF from LaTeX source files:

```bash
cd latex/
pdflatex -interaction=nonstopmode LLM_Study_Material.tex
pdflatex -interaction=nonstopmode LLM_Study_Material.tex  # Run twice for TOC
```

The compiled PDF will be generated in the `latex/` directory and can be moved to `docs/` for distribution.

## Contributing

This is a study materials repository. If you identify errors or areas for improvement, please document them clearly with:
- Section/page reference
- Description of the issue
- Suggested correction with supporting references

## License

These materials are provided for educational purposes. Please respect copyright and attribution requirements when using external code libraries or datasets referenced in the notebooks.

## Acknowledgments

This study guide synthesizes knowledge from research papers, technical documentation, and open-source implementations. Specific citations are included throughout the PDF document where applicable.

---

## Citation

If you use these materials in your research or teaching, please cite:

```bibtex
@misc{genai_study_materials_2025,
  title={Comprehensive Study Materials for Large Language Models and Generative AI},
  author={Ayana},
  year={2025},
  publisher={GitHub},
  howpublished={\\url{https://github.com/YOUR_USERNAME/GenAi_Prep}}
}
```

## Related Topics

`large-language-models` `transformers` `deep-learning` `machine-learning` `nlp` `natural-language-processing` `artificial-intelligence` `rlhf` `fine-tuning` `lora` `qlora` `gpt` `bert` `attention-mechanism` `retrieval-augmented-generation` `langchain` `langgraph` `pre-training` `instruction-tuning` `reinforcement-learning` `python` `jupyter-notebook` `education` `study-guide` `university` `research`
