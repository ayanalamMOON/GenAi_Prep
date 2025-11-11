"""
Visualization 6 of 8 for Section 10: Error Analysis and Model Failures

Output: visualization/images/evaluation/error_analysis.png
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook", font_scale=1.1)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'

OUTPUT_DIR = "../images/evaluation"

print("SECTION 10 - VISUALIZATION 6/8: Error Analysis")

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# Error type distribution
ax1 = fig.add_subplot(gs[0, :2])
error_types = ['Factual\nErrors', 'Hallucin-\nations', 'Reasoning\nFailures', 'Context\nMisunder-\nstanding', 'Unsafe\nContent', 'Format\nErrors']
base_errors = [28.5, 35.2, 42.1, 31.8, 15.3, 12.7]
finetuned_errors = [12.3, 15.8, 22.4, 14.2, 5.1, 6.8]

x = np.arange(len(error_types))
width = 0.35
ax1.bar(x - width/2, base_errors, width, label='Base Model', color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.2)
ax1.bar(x + width/2, finetuned_errors, width, label='Fine-tuned', color='#27ae60', alpha=0.85, edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Error Rate (%)', fontweight='bold', fontsize=12)
ax1.set_title('Error Type Distribution', fontweight='bold', fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(error_types, fontsize=10)
ax1.legend(frameon=True, fancybox=True, shadow=True)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Severity breakdown
ax2 = fig.add_subplot(gs[0, 2])
severities = ['Critical', 'Major', 'Minor']
counts = [42, 138, 285]
colors_sev = ['#e74c3c', '#f39c12', '#3498db']
wedges, texts, autotexts = ax2.pie(counts, labels=severities, autopct='%1.1f%%', colors=colors_sev, 
                                     startangle=90, explode=(0.1, 0.05, 0), shadow=True)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)
ax2.set_title('Error Severity Distribution', fontweight='bold', fontsize=13)

# Error by input length
ax3 = fig.add_subplot(gs[1, :2])
input_lengths = ['0-50\ntokens', '51-100\ntokens', '101-200\ntokens', '201-500\ntokens', '500+\ntokens']
error_rates = [8.2, 12.5, 18.3, 24.7, 35.2]
ax3.plot(range(len(input_lengths)), error_rates, 'o-', linewidth=3, markersize=12, 
         color='#e74c3c', markeredgecolor='black', markeredgewidth=1.5)
ax3.fill_between(range(len(input_lengths)), error_rates, alpha=0.3, color='#e74c3c')
ax3.set_xlabel('Input Length', fontweight='bold', fontsize=12)
ax3.set_ylabel('Error Rate (%)', fontweight='bold', fontsize=12)
ax3.set_title('Error Rate vs Input Length', fontweight='bold', fontsize=13)
ax3.set_xticks(range(len(input_lengths)))
ax3.set_xticklabels(input_lengths, fontsize=10)
ax3.grid(True, alpha=0.3, linestyle='--')

# Failure modes
ax4 = fig.add_subplot(gs[1, 2])
modes = ['Refuses\nvalid\ntask', 'Overcon-\nfident', 'Verbose', 'Repetitive']
mode_freq = [8.5, 22.3, 15.8, 12.4]
ax4.barh(modes, mode_freq, color='#9b59b6', alpha=0.85, edgecolor='black', linewidth=1.2)
ax4.set_xlabel('Frequency (%)', fontweight='bold', fontsize=12)
ax4.set_title('Common Failure Modes', fontweight='bold', fontsize=13)
ax4.grid(axis='x', alpha=0.3, linestyle='--')

plt.suptitle('Error Analysis: Types, Severity, Input Length, Failure Modes', fontsize=15, fontweight='bold', y=0.995)
plt.savefig(os.path.join(OUTPUT_DIR, "error_analysis.png"), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: error_analysis.png\nVISUALIZATION 6/8 COMPLETE")
