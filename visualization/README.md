# LLM Study Material Visualizations

This directory contains Python scripts to generate visualizations for the LLM Study Material LaTeX document.

## Directory Structure

```
visualization/
├── scripts/                          # Python visualization scripts
│   └── fine_tuning_visualizations.py # Fine-tuning section visualizations
├── images/                           # Generated visualization images
│   └── fine_tuning/                  # Fine-tuning specific images
│       ├── transfer_learning_process.png
│       ├── learning_rate_comparison.png
│       ├── discriminative_learning_rates.png
│       ├── gradient_flow_analysis.png
│       ├── catastrophic_forgetting.png
│       ├── fisher_information_matrix.png
│       ├── training_loss_curves.png
│       ├── memory_comparison.png
│       ├── hyperparameter_recommendations.png
│       └── convergence_analysis.png
└── README.md                         # This file
```

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

### Fine-Tuning Visualizations

```bash
cd visualization/scripts
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

To add visualizations for other sections (LoRA, QLoRA, LangChain, etc.):

1. Create a new script in `scripts/` (e.g., `lora_visualizations.py`)
2. Create corresponding image directory (e.g., `images/lora/`)
3. Follow the pattern from `fine_tuning_visualizations.py`:
   - Import matplotlib, seaborn, numpy
   - Define output directory
   - Create functions for each visualization
   - Save to appropriate directory with descriptive names
4. Update LaTeX to include new images with `\includegraphics{}`

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
