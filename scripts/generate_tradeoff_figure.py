#!/usr/bin/env python3
"""
Generate sample efficiency vs final performance trade-off figure
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set publication-quality defaults
plt.rcParams['figure.figsize'] = (6, 4.5)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

# Data from convergence analysis (Zork1 only)
methods = {
    'Static Shaping': {'steps': 2933, 'score': 27.1, 'color': '#666666', 'marker': 's'},
    'No Shaping': {'steps': 5566, 'score': 30.7, 'color': '#999999', 'marker': 'o'},
    'Exponential': {'steps': 3500, 'score': 27.3, 'color': '#1f77b4', 'marker': '^'},
    'Linear': {'steps': 7766, 'score': 32.9, 'color': '#ff7f0e', 'marker': 'v'},
    'Cosine': {'steps': 6700, 'score': 33.8, 'color': '#2ca02c', 'marker': 'D'},
    'Sparsity-Triggered': {'steps': 16033, 'score': 41.6, 'color': '#d62728', 'marker': '*'},
    'Sparsity-Sensitive': {'steps': 23233, 'score': 36.9, 'color': '#9467bd', 'marker': 'P'},
    'Entropy-Informed': {'steps': 22866, 'score': 34.7, 'color': '#8c564b', 'marker': 'X'},
}

fig, ax = plt.subplots(figsize=(7, 5))

# Plot each method
for name, data in methods.items():
    if 'Shaping' in name or 'No Shaping' in name:
        # Baselines
        ax.scatter(data['steps'], data['score'], 
                  color=data['color'], marker=data['marker'], 
                  s=150, alpha=0.8, edgecolors='black', linewidths=1.5,
                  label=name, zorder=3)
    else:
        # Adaptive methods
        ax.scatter(data['steps'], data['score'], 
                  color=data['color'], marker=data['marker'], 
                  s=150, alpha=0.9, edgecolors='black', linewidths=1,
                  label=name, zorder=4)

# Add trend line
steps_array = np.array([d['steps'] for d in methods.values()])
scores_array = np.array([d['score'] for d in methods.values()])
z = np.polyfit(steps_array, scores_array, 1)
p = np.poly1d(z)
x_line = np.linspace(steps_array.min(), steps_array.max(), 100)
ax.plot(x_line, p(x_line), 'k--', alpha=0.3, linewidth=1.5, zorder=1)

# Annotations for key points
ax.annotate('Fast but poor\nfinal performance', 
            xy=(2933, 27.1), xytext=(5000, 24),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1),
            fontsize=9, ha='left', color='gray')

ax.annotate('Slow but best\nfinal performance', 
            xy=(16033, 41.6), xytext=(18000, 44),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1),
            fontsize=9, ha='left', color='gray')

ax.set_xlabel('Steps to Reach 80% of Final Performance', fontsize=12, fontweight='bold')
ax.set_ylabel('Final Score (50k steps)', fontsize=12, fontweight='bold')
ax.set_title('Sample Efficiency vs Final Performance Trade-off (Zork1)', 
             fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='lower right', framealpha=0.95, ncol=2, fontsize=8)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_xlim(0, 25000)
ax.set_ylim(24, 45)

plt.tight_layout()

# Save
output_dir = Path('results/figures')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'efficiency_tradeoff.pdf', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'efficiency_tradeoff.png', bbox_inches='tight', dpi=300)
print("Saved efficiency_tradeoff.pdf/png")
plt.close()
