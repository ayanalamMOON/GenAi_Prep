"""
Visualization 8 of 8 for Section 10: Evaluation Best Practices Summary

Output: visualization/images/evaluation/evaluation_best_practices.png
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook", font_scale=1.1)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'

OUTPUT_DIR = "../../images/evaluation"

print("SECTION 10 - VISUALIZATION 8/8: Evaluation Best Practices")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

# Evaluation pipeline stages
ax1 = fig.add_subplot(gs[0, :])
stages = ['Data\nCollection', 'Automatic\nMetrics', 'Benchmark\nEvaluation', 'Human\nEvaluation', 'Error\nAnalysis', 'Production\nMonitoring']
importance = [8, 7, 9, 10, 8, 9]
cost = [4, 2, 3, 9, 6, 7]
x = np.arange(len(stages))
width = 0.35
ax1.bar(x - width/2, importance, width, label='Importance (0-10)', color='#27ae60', alpha=0.85, edgecolor='black', linewidth=1.2)
ax1.bar(x + width/2, cost, width, label='Cost/Effort (0-10)', color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Score (0-10)', fontweight='bold', fontsize=12)
ax1.set_title('Evaluation Pipeline: Importance vs Cost Trade-off', fontweight='bold', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(stages, fontsize=11)
ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=12)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, 12)

# Metric selection guide
ax2 = fig.add_subplot(gs[1, 0])
scenarios = ['Research\nPaper', 'Production\nDeployment', 'Internal\nDev', 'Domain\nAdaptation']
use_perplexity = [9, 6, 10, 8]
use_bleu = [8, 5, 7, 7]
use_human = [10, 10, 6, 8]
use_benchmarks = [10, 9, 7, 9]
x = np.arange(len(scenarios))
width = 0.2
ax2.bar(x - 1.5*width, use_perplexity, width, label='Perplexity', color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.2)
ax2.bar(x - 0.5*width, use_bleu, width, label='BLEU/ROUGE', color='#9b59b6', alpha=0.85, edgecolor='black', linewidth=1.2)
ax2.bar(x + 0.5*width, use_human, width, label='Human Eval', color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.2)
ax2.bar(x + 1.5*width, use_benchmarks, width, label='Benchmarks', color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1.2)
ax2.set_ylabel('Usefulness (0-10)', fontweight='bold', fontsize=12)
ax2.set_title('Metric Selection by Use Case', fontweight='bold', fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels(scenarios, fontsize=10)
ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, 12)

# Common pitfalls
ax3 = fig.add_subplot(gs[1, 1])
pitfalls = ['Only\nAutomatic\nMetrics', 'Small\nSample\nSize', 'No\nHuman\nBaseline', 'Ignoring\nEdge Cases', 'Single\nMetric\nFocus']
severity = [8.5, 7.2, 6.8, 9.1, 8.8]
colors_pit = ['#e74c3c' if s > 8 else '#f39c12' if s > 7 else '#3498db' for s in severity]
bars = ax3.bar(pitfalls, severity, color=colors_pit, alpha=0.85, edgecolor='black', linewidth=1.2)
ax3.set_ylabel('Severity Score (0-10)', fontweight='bold', fontsize=12)
ax3.set_title('Common Evaluation Pitfalls', fontweight='bold', fontsize=13)
ax3.set_ylim(0, 10)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.tick_params(axis='x', labelsize=9)
for bar, s in zip(bars, severity):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.3, f'{s:.1f}',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.suptitle('Evaluation Best Practices: Pipeline, Metric Selection, Common Pitfalls',
             fontsize=15, fontweight='bold', y=0.995)
plt.savefig(os.path.join(OUTPUT_DIR, "evaluation_best_practices.png"), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: evaluation_best_practices.png\nVISUALIZATION 8/8 COMPLETE")
print("\n" + "="*80)
print("ALL SECTION 10 VISUALIZATIONS COMPLETE!")
print("="*80)
