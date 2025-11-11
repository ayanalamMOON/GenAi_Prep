"""
Visualization 7 of 8 for Section 10: Multi-Task Evaluation Dashboard

Output: visualization/images/evaluation/multitask_evaluation.png
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook", font_scale=1.1)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'

OUTPUT_DIR = "../images/evaluation"

print("SECTION 10 - VISUALIZATION 7/8: Multi-Task Evaluation")

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# Task performance heatmap
ax1 = fig.add_subplot(gs[:, :2])
tasks = ['QA', 'Summ.', 'Trans.', 'Code', 'Math', 'Reason.', 'Dialog']
models = ['Base', 'FT-1K', 'FT-10K', 'FT-100K', 'Specialist', 'Ensemble']
np.random.seed(42)
perf_data = np.array([
    [45, 42, 48, 38, 41, 46, 44],
    [58, 54, 61, 52, 55, 59, 57],
    [68, 66, 72, 64, 67, 70, 69],
    [75, 73, 78, 72, 74, 77, 76],
    [62, 88, 65, 92, 91, 68, 64],
    [78, 76, 80, 75, 77, 79, 78]
])

im = ax1.imshow(perf_data, cmap='YlGnBu', aspect='auto', vmin=30, vmax=95)
ax1.set_xticks(np.arange(len(tasks)))
ax1.set_yticks(np.arange(len(models)))
ax1.set_xticklabels(tasks, fontsize=11)
ax1.set_yticklabels(models, fontsize=11)
ax1.set_title('Multi-Task Performance Heatmap (%)', fontweight='bold', fontsize=14)
for i in range(len(models)):
    for j in range(len(tasks)):
        ax1.text(j, i, f'{perf_data[i,j]}', ha='center', va='center',
                color='white' if perf_data[i,j] > 65 else 'black', fontsize=10, fontweight='bold')
cbar = plt.colorbar(im, ax=ax1, fraction=0.03, pad=0.02)
cbar.set_label('Accuracy (%)', fontweight='bold')

# Average performance
ax2 = fig.add_subplot(gs[0, 2])
avg_scores = perf_data.mean(axis=1)
ax2.barh(models, avg_scores, color=['#e74c3c', '#f39c12', '#2ecc71', '#27ae60', '#9b59b6', '#3498db'],
         alpha=0.85, edgecolor='black', linewidth=1.2)
ax2.set_xlabel('Avg Score (%)', fontweight='bold')
ax2.set_title('Average Performance', fontweight='bold', fontsize=13)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
for i, v in enumerate(avg_scores):
    ax2.text(v + 1, i, f'{v:.1f}', va='center', fontweight='bold')

# Task difficulty ranking
ax3 = fig.add_subplot(gs[1, 2])
task_avg = perf_data.mean(axis=0)
sorted_idx = np.argsort(task_avg)
colors_diff = ['#e74c3c' if v < 60 else '#f39c12' if v < 70 else '#27ae60' for v in task_avg[sorted_idx]]
ax3.barh([tasks[i] for i in sorted_idx], task_avg[sorted_idx], color=colors_diff,
         alpha=0.85, edgecolor='black', linewidth=1.2)
ax3.set_xlabel('Avg Score (%)', fontweight='bold')
ax3.set_title('Task Difficulty Ranking', fontweight='bold', fontsize=13)
ax3.grid(axis='x', alpha=0.3, linestyle='--')

plt.suptitle('Multi-Task Evaluation: Performance Heatmap, Averages, Task Difficulty',
             fontsize=15, fontweight='bold', y=0.995)
plt.savefig(os.path.join(OUTPUT_DIR, "multitask_evaluation.png"), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: multitask_evaluation.png\nVISUALIZATION 7/8 COMPLETE")
