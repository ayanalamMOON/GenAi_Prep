# AI Coding Agent Instructions for GenAi_Prep Repository

## Repository Purpose

This is an **educational materials repository** for university-level LLM study, not a production codebase. The primary artifact is a 335+ page LaTeX study guide (`latex/LLM_Study_Material.tex`) with supporting Python visualization scripts and Jupyter notebooks. Content is **actively maintained** with regular enhancements following a specific pedagogical philosophy.

## Architecture & Component Relationships

### Core Components

1. **LaTeX Study Guide** (`latex/LLM_Study_Material.tex`, ~18,681 lines)
   - Single monolithic document covering 12 major sections
   - Compiled with `pdflatex` (run twice for TOC: `pdflatex -interaction=nonstopmode LLM_Study_Material.tex`)
   - Output: `docs/LLM_Study_Material.pdf` (335+ pages, 15.93 MB)
   - Uses custom `tcolorbox` environments for structured content (see conventions below)

2. **Visualization System** (`visualization/`)
   - 6 Python scripts generate 47 PNG visualizations (300 DPI, publication quality)
   - Scripts: `fine_tuning_visualizations.py`, `lora_visualizations.py`, `qlora_visualizations.py`, `pretraining_visualizations.py`, `langchain_visualizations.py`, `langgraph_visualizations.py`
   - Output: `visualization/images/{category}/` directories
   - Dependencies: matplotlib, seaborn, numpy (NOT in requirements.txt - install separately)

3. **Jupyter Notebooks** (`notebooks/`, 4 files)
   - Practical implementations demonstrating study guide concepts
   - Self-contained with inline library installations (`!pip install`)
   - Focus: Fine-tuning, LoRA, QLoRA, pre-training workflows

## Critical Workflows

### LaTeX Document Enhancement (Primary Workflow)

**When adding/enhancing sections, follow the established pattern from Section 8 (RLHF):**

1. **Content Structure** (teacher-quality standard):
   - Start with intuition before formalization
   - Include step-by-step mathematical derivations with clear notation
   - Add production-ready code with extensive inline comments
   - Explain code in accompanying text (never just dump code)

2. **Required Components for Each Subsection**:
   ```latex
   % Hyperparameter Guidance Box
   \begin{tcolorbox}[colback=blue!5!white,colframe=blue!75!black,title={Hyperparameter Guidance: Topic Name}]
   \textbf{Recommended Values:}
   \begin{itemize}
       \item Parameter: value range (justification with compute/memory impact)
   \end{itemize}
   \end{tcolorbox}

   % Common Pitfalls Box
   \begin{tcolorbox}[colback=orange!5!white,colframe=orange!75!black,title={Common Pitfalls and Debugging Tips: Topic Name}]
   \textbf{Pitfall Name:}
   \begin{itemize}
       \item \textbf{Symptoms:} Observable behavior
       \item \textbf{Root Cause:} Why it happens
       \item \textbf{Solution:} Step-by-step fix with code snippets
       \item \textbf{Prevention:} How to avoid
   \end{itemize}
   \end{tcolorbox}

   % Key Takeaways Box
   \begin{tcolorbox}[colback=green!5!white,colframe=green!75!black,title={Key Takeaways: Topic Name}]
   \begin{enumerate}
       \item Core equation or concept with brief explanation
       \item Success indicators (e.g., "Loss should converge within 500 steps")
       \item Critical implementation detail
   \end{enumerate}
   \end{tcolorbox}
   ```

3. **tcolorbox Title Syntax** (CRITICAL - avoid rendering errors):
   - Wrap titles in braces: `title={Title Text}`
   - Never use `\textbf{}` inside title parameter
   - Replace `\&` with "and" (e.g., "Common Pitfalls and Debugging Tips")
   - Examples in lines 9251, 9277, 9308, 9434, 9460, 9499, 9795, 9852, 9942, 10678, 10966, 11013, 11085

4. **Code Documentation Standards**:
   - Production code examples: 400-700 lines per implementation
   - Inline comments explaining every significant operation
   - Accompanying text blocks explaining "why" before code
   - Real hyperparameter values with justifications (e.g., `learning_rate=2e-5  # Prevents catastrophic forgetting`)
   - Step-by-step numbered structure (Step 1: Setup, Step 2: Configuration, etc.)

### Visualization Generation

**When creating new visualizations:**

1. Create script in `visualization/scripts/{topic}_visualizations.py`
2. Follow pattern from `fine_tuning_visualizations.py`:
   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   import seaborn as sns
   plt.style.use('seaborn-v0_8-darkgrid')
   OUTPUT_DIR = "../images/{topic}"
   os.makedirs(OUTPUT_DIR, exist_ok=True)
   # Each visualization as separate function
   # Save with: plt.savefig(f"{OUTPUT_DIR}/descriptive_name.png", dpi=300, bbox_inches='tight')
   ```
3. Create corresponding `visualization/images/{topic}/` directory
4. Reference in LaTeX: `\includegraphics[width=0.95\textwidth]{visualization/images/{topic}/{name}.png}`
5. Update `visualization/README.md` with generation instructions

**Run all visualization scripts:**
```bash
cd visualization/scripts
python fine_tuning_visualizations.py  # Generates 10 images
python lora_visualizations.py         # Generates 9 images
python qlora_visualizations.py        # Generates 6 images
python pretraining_visualizations.py  # Generates 8 images
python langchain_visualizations.py    # Generates 6 images
python langgraph_visualizations.py    # Generates 8 images
```

### LaTeX Compilation

**Critical compilation steps:**
```bash
cd latex/
pdflatex -interaction=nonstopmode LLM_Study_Material.tex  # First pass
pdflatex -interaction=nonstopmode LLM_Study_Material.tex  # Second pass for TOC
# Output: LLM_Study_Material.pdf, .aux, .log, .out, .toc files
mv LLM_Study_Material.pdf ../docs/  # Move to docs for distribution
```

**Common compilation issues:**
- Missing `\usepackage{tcolorbox}` in preamble (line ~10-15)
- Image paths must be relative to LaTeX document root: `visualization/images/...`
- Verify image existence before referencing: `ls -la visualization/images/{category}/`

## Project-Specific Conventions

### Content Enhancement Philosophy (5-Point Standard)

1. **Teacher-Quality Exposition**: First principles → Intuition → Formalization → Implementation
2. **Mathematical Rigor**: Step-by-step derivations, clear notation, dimensionality analysis
3. **Production Code**: Extensively commented, runnable, real hyperparameters
4. **Practical Guidance**: Hyperparameter recommendations, pitfalls, debugging
5. **Structured Summaries**: Key takeaways for quick reference

### Naming Conventions

- **LaTeX sections**: Numbered sequentially (Section 1: Fine-Tuning, Section 8: RLHF, etc.)
- **Visualization files**: `snake_case` descriptive names (e.g., `gradient_flow_analysis.png`)
- **Notebooks**: `PascalCase` with underscores (e.g., `GPT_2_FINETUNING_QLoRA.ipynb`)
- **Git commits**: Detailed multi-line format with summary, bullet points, context (see commit 3379773)

### Tone & Style

- **No emojis** anywhere (professional academic tone)
- **No AI slop** (avoid "super positive" language like "Exciting!", "Amazing!")
- Professional, clear, factual writing suitable for university-level students
- Example bad: "🚀 This is amazing! Let's dive in!" 
- Example good: "This section covers the mathematical foundations of attention mechanisms."

## Current Development Status

### Recently Completed (v1.0.0)
- Section 8 (RLHF) enhanced with ~900 lines: TRL PPO implementation (~725 lines), DPO walkthrough, 13 tcolorbox instances
- All tcolorbox rendering issues fixed (proper title syntax)
- Repository structure reorganized (docs/, latex/, notebooks/, visualization/)

### In Progress (see README "Development Status & Roadmap")
- Section 9: Instruction Fine-Tuning & SFT Datasets
- Section 10: Evaluation & Benchmarking  
- Section 11: Safety, Ethics & Bias Mitigation

### Update Frequency
- Major updates: Monthly (new sections, significant enhancements)
- Minor updates: Bi-weekly (fixes, clarifications, examples)
- Continuous: Bug fixes, typo corrections

## Git Workflow

**Repository**: https://github.com/ayanalamMOON/GenAi_Prep

**When making changes:**
```bash
git add .
git commit -m "Brief summary

- Detailed change 1 with context
- Detailed change 2 with file references
- Impact/rationale

Technical notes or cross-references"
git push origin main
```

**For version tags:**
```bash
git tag -a v1.x.0 -m "Version 1.x.0: Description
Features:
- Feature 1
- Feature 2
Content changes: specific sections modified"
git push origin --tags
```

## External Dependencies

- **LaTeX**: Full distribution required (MiKTeX on Windows, TeX Live on Linux/Mac)
- **Python**: 3.8+ (not in requirements.txt, install globally or in venv)
- **Visualization libs**: `pip install matplotlib seaborn numpy` (separate from requirements.txt)
- **Notebook libs**: Auto-installed via `!pip install` cells (transformers, datasets, peft, accelerate, bitsandbytes)
- **Git**: Required for version control and GitHub CLI (`gh`)

## Key Files Reference

- **Main document**: `latex/LLM_Study_Material.tex` (18,681 lines, Sections 1-12)
- **Enhanced section example**: Lines 9169-11487 (Section 8 RLHF - reference for quality standard)
- **Visualization patterns**: `visualization/scripts/fine_tuning_visualizations.py` (580 lines)
- **Repository structure**: Root `README.md` (374 lines with roadmap)
- **LaTeX preamble**: `latex/LLM_Study_Material.tex` lines 1-70 (packages, custom commands)

## Important Don'ts

1. **Never** add emojis to any file (README, commits, code comments)
2. **Never** use `\textbf{}` or `\&` in tcolorbox title parameters
3. **Never** write code without extensive explanatory text
4. **Never** update requirements.txt for visualization libraries (kept minimal intentionally)
5. **Never** create separate summary MD files unless explicitly requested
6. **Never** assume LaTeX will auto-break long sections - use manual page breaks if needed

## Testing & Validation

**After LaTeX changes:**
```bash
cd latex/
pdflatex -interaction=nonstopmode LLM_Study_Material.tex
# Check output: No errors, page count increased appropriately, new content renders
wc -l LLM_Study_Material.tex  # Verify line count
```

**After visualization changes:**
```bash
cd visualization/scripts
python {script_name}.py
ls -la ../images/{category}/  # Verify new images created
file ../images/{category}/*.png  # Check image properties (should show PNG, 300 DPI)
```

**Before pushing to GitHub:**
```bash
git status  # Verify intended files staged
git log --oneline -3  # Check commit messages follow convention
gh repo view  # Verify remote connection
```
