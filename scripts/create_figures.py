#!/usr/bin/env python3
"""
Publication-Quality Figure Generation
Creates professional, high-impact visualizations for research paper
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

# Publication-quality settings
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'legend.framealpha': 0.95,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.5,
    'lines.markersize': 6,
})

# Professional color palette (colorblind-friendly)
COLORS = {
    'baseline_no': '#7f7f7f',      # Gray
    'baseline_static': '#2c2c2c',  # Dark gray
    'cosine': '#1f77b4',           # Blue
    'exponential': '#ff7f0e',      # Orange
    'linear': '#bcbd22',           # Yellow-green
    'sparsity_50': '#d62728',      # Red
    'sparsity_25': '#9467bd',      # Purple
    'uncertainty': '#8c564b',      # Brown
}

EXPERIMENT_DIR = "logging/final/zork1/seed0"
OUTPUT_DIR = "results/figures/zork1/seed0"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Method configurations
METHODS = {
    'baseline_no_shaping': {
        'name': 'SAC Baseline',
        'short': 'Baseline',
        'color': COLORS['baseline_no'],
        'linestyle': '--',
        'linewidth': 3,
        'alpha': 0.9,
        'zorder': 1,
        'category': 'baseline'
    },
    'baseline_static_shaping': {
        'name': 'Static Shaping',
        'short': 'Static',
        'color': COLORS['baseline_static'],
        'linestyle': ':',
        'linewidth': 3,
        'alpha': 0.9,
        'zorder': 2,
        'category': 'baseline'
    },
    'adaptive_time_decay_cosine': {
        'name': 'Cosine Decay',
        'short': 'Cosine',
        'color': COLORS['cosine'],
        'linestyle': '-',
        'linewidth': 2.5,
        'alpha': 1.0,
        'zorder': 10,
        'category': 'adaptive'
    },
    'adaptive_time_decay_exp': {
        'name': 'Exponential Decay',
        'short': 'Exponential',
        'color': COLORS['exponential'],
        'linestyle': '-',
        'linewidth': 2.5,
        'alpha': 0.85,
        'zorder': 8,
        'category': 'adaptive'
    },
    'adaptive_time_decay_linear': {
        'name': 'Linear Decay',
        'short': 'Linear',
        'color': COLORS['linear'],
        'linestyle': '-',
        'linewidth': 2.5,
        'alpha': 0.85,
        'zorder': 7,
        'category': 'adaptive'
    },
    'adaptive_sparsity_triggered': {
        'name': 'Sparsity-Triggered (τ=50)',
        'short': 'Sparsity-50',
        'color': COLORS['sparsity_50'],
        'linestyle': '-',
        'linewidth': 2.5,
        'alpha': 1.0,
        'zorder': 9,
        'category': 'adaptive'
    },
    'adaptive_sparsity_sensitive': {
        'name': 'Sparsity-Sensitive (τ=25)',
        'short': 'Sparsity-25',
        'color': COLORS['sparsity_25'],
        'linestyle': '-',
        'linewidth': 2.5,
        'alpha': 1.0,
        'zorder': 9,
        'category': 'adaptive'
    },
    'adaptive_uncertainty_informed': {
        'name': 'Uncertainty-Informed',
        'short': 'Uncertainty',
        'color': COLORS['uncertainty'],
        'linestyle': '-',
        'linewidth': 2.5,
        'alpha': 0.7,
        'zorder': 3,
        'category': 'adaptive'
    }
}


def load_data():
    """Load all experimental data."""
    data = {}
    for method_dir in Path(EXPERIMENT_DIR).iterdir():
        if not method_dir.is_dir():
            continue
        
        method_name = method_dir.name
        progress_file = method_dir / "progress.json"
        
        if not progress_file.exists():
            continue
        
        scores = []
        steps = []
        
        with open(progress_file, 'r') as f:
            for i, line in enumerate(f):
                try:
                    entry = json.loads(line.strip())
                    scores.append(entry['train/Last100EpisodeScores'])
                    steps.append((i + 1) * 100)
                except:
                    continue
        
        if scores:
            data[method_name] = {
                'steps': np.array(steps),
                'scores': np.array(scores),
                'final_score': scores[-1],
                'max_score': max(scores),
                'completed': len(scores) >= 200
            }
    
    return data


def smooth_curve(y, sigma=2):
    """Apply Gaussian smoothing to curve."""
    return gaussian_filter1d(y, sigma=sigma)


def create_learning_curves(data):
    """Create publication-quality learning curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort methods: baselines first, then adaptive by final score
    baseline_methods = [m for m in data.keys() if METHODS.get(m, {}).get('category') == 'baseline']
    adaptive_methods = [m for m in data.keys() if METHODS.get(m, {}).get('category') == 'adaptive']
    adaptive_methods.sort(key=lambda m: data[m]['final_score'], reverse=True)
    
    methods_ordered = baseline_methods + adaptive_methods
    
    # Plot each method
    for method in methods_ordered:
        if method not in METHODS:
            continue
        
        config = METHODS[method]
        method_data = data[method]
        
        # Smooth the curve for better visualization
        scores_smooth = smooth_curve(method_data['scores'], sigma=1.5)
        
        ax.plot(method_data['steps'], scores_smooth,
                label=config['name'],
                color=config['color'],
                linestyle=config['linestyle'],
                linewidth=config['linewidth'],
                alpha=config['alpha'],
                zorder=config['zorder'])
    
    # Styling
    ax.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Score (Last 100 Episodes)', fontsize=13, fontweight='bold')
    ax.set_title('Learning Curves: Adaptive vs Static Reward Shaping', 
                 fontsize=15, fontweight='bold', pad=15)
    
    # Grid
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Legend with two columns
    legend = ax.legend(loc='lower right', ncol=2, frameon=True, 
                      fancybox=False, shadow=False, 
                      columnspacing=1.0, handlelength=2.5)
    legend.get_frame().set_linewidth(1.2)
    legend.get_frame().set_edgecolor('black')
    
    # Set limits
    ax.set_xlim(0, 50000)
    ax.set_ylim(0, max([d['max_score'] for d in data.values()]) * 1.05)
    
    # Spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/learning_curves.png", bbox_inches='tight', facecolor='white')
    plt.savefig(f"{OUTPUT_DIR}/learning_curves.pdf", bbox_inches='tight', facecolor='white')
    print(f"Created: learning_curves.png/pdf")
    plt.close()


def create_final_performance_comparison(data):
    """Create elegant final performance comparison."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Prepare data sorted by score
    methods_sorted = sorted(data.items(), key=lambda x: x[1]['final_score'], reverse=True)
    
    names = []
    scores = []
    colors = []
    
    for method, method_data in methods_sorted:
        if method not in METHODS:
            continue
        names.append(METHODS[method]['name'])
        scores.append(method_data['final_score'])
        colors.append(METHODS[method]['color'])
    
    y_pos = np.arange(len(names))
    
    # Create horizontal bars
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.85, 
                   edgecolor='black', linewidth=1.5, height=0.7)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.5, i, f'{score:.1f}', 
                va='center', ha='left', fontsize=11, fontweight='bold')
    
    # Add baseline reference line
    baseline_score = data['baseline_no_shaping']['final_score']
    ax.axvline(x=baseline_score, color='red', linestyle='--', 
               linewidth=2, alpha=0.6, zorder=0, label=f'Baseline ({baseline_score:.1f})')
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('Final Score (Last 100 Episodes)', fontsize=13, fontweight='bold')
    ax.set_title('Final Performance Comparison', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(0, max(scores) * 1.12)
    
    # Grid
    ax.grid(True, axis='x', alpha=0.25, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='lower right', frameon=True, fancybox=False, shadow=False)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.2)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/final_performance.png", bbox_inches='tight', facecolor='white')
    plt.savefig(f"{OUTPUT_DIR}/final_performance.pdf", bbox_inches='tight', facecolor='white')
    print(f"Created: final_performance.png/pdf")
    plt.close()


def create_improvement_heatmap(data):
    """Create improvement percentage visualization."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    baseline_score = data['baseline_no_shaping']['final_score']
    
    # Prepare data
    methods_sorted = sorted(data.items(), key=lambda x: x[1]['final_score'], reverse=True)
    
    names = []
    improvements = []
    colors_list = []
    
    for method, method_data in methods_sorted:
        if method == 'baseline_no_shaping' or method not in METHODS:
            continue
        
        improvement = ((method_data['final_score'] - baseline_score) / baseline_score) * 100
        names.append(METHODS[method]['name'])
        improvements.append(improvement)
        
        # Color based on improvement
        if improvement > 25:
            colors_list.append('#2ca02c')  # Green
        elif improvement > 15:
            colors_list.append('#1f77b4')  # Blue
        elif improvement > 0:
            colors_list.append('#ff7f0e')  # Orange
        else:
            colors_list.append('#d62728')  # Red
    
    y_pos = np.arange(len(names))
    
    # Create bars
    bars = ax.barh(y_pos, improvements, color=colors_list, alpha=0.85,
                   edgecolor='black', linewidth=1.5, height=0.7)
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        label_x = imp + 1 if imp > 0 else imp - 1
        ha = 'left' if imp > 0 else 'right'
        ax.text(label_x, i, f'{imp:+.1f}%', 
                va='center', ha=ha, fontsize=11, fontweight='bold')
    
    # Add zero line
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8, zorder=0)
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('Improvement over Baseline (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Performance Improvement (Baseline: {baseline_score:.1f})', 
                 fontsize=15, fontweight='bold', pad=15)
    
    # Grid
    ax.grid(True, axis='x', alpha=0.25, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.2)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/improvement_analysis.png", bbox_inches='tight', facecolor='white')
    plt.savefig(f"{OUTPUT_DIR}/improvement_analysis.pdf", bbox_inches='tight', facecolor='white')
    print(f"Created: improvement_analysis.png/pdf")
    plt.close()


def create_category_performance(data):
    """Create elegant category comparison with violin plots."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Organize by category
    categories = {
        'Baseline\n(n=2)': [],
        'Time-Based\n(n=3)': [],
        'Sparsity-Based\n(n=2)': [],
    }
    
    for method, method_data in data.items():
        if method not in METHODS:
            continue
        
        category = METHODS[method]['category']
        if category == 'baseline':
            categories['Baseline\n(n=2)'].append(method_data['final_score'])
        elif category == 'adaptive':
            if 'time_decay' in method:
                categories['Time-Based\n(n=3)'].append(method_data['final_score'])
            elif 'sparsity' in method:
                categories['Sparsity-Based\n(n=2)'].append(method_data['final_score'])
    
    # Prepare data
    cat_names = list(categories.keys())
    cat_data = [categories[name] for name in cat_names]
    
    # Create box plots with custom styling
    bp = ax.boxplot(cat_data, labels=cat_names, patch_artist=True,
                    widths=0.6,
                    boxprops=dict(facecolor='lightblue', alpha=0.7, linewidth=1.5),
                    medianprops=dict(color='darkred', linewidth=3),
                    whiskerprops=dict(linewidth=1.5, color='black'),
                    capprops=dict(linewidth=1.5, color='black'),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=8, 
                                   linestyle='none', markeredgecolor='black'))
    
    # Add individual points with jitter
    for i, scores in enumerate(cat_data):
        x = np.random.normal(i+1, 0.04, size=len(scores))
        ax.scatter(x, scores, alpha=0.8, s=120, color='darkblue', 
                  edgecolors='black', linewidth=1.5, zorder=10)
        
        # Add mean marker
        mean_val = np.mean(scores)
        ax.scatter(i+1, mean_val, marker='D', s=150, color='gold', 
                  edgecolors='black', linewidth=2, zorder=11, label='Mean' if i == 0 else '')
    
    # Styling
    ax.set_ylabel('Final Score', fontsize=13, fontweight='bold')
    ax.set_title('Performance by Scheduler Category', fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim(0, max([max(d) for d in cat_data]) * 1.1)
    
    # Grid
    ax.grid(True, axis='y', alpha=0.25, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper left', frameon=True, fancybox=False, shadow=False)
    
    # Spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/category_performance.png", bbox_inches='tight', facecolor='white')
    plt.savefig(f"{OUTPUT_DIR}/category_performance.pdf", bbox_inches='tight', facecolor='white')
    print(f"Created: category_performance.png/pdf")
    plt.close()


def create_summary_figure(data):
    """Create a comprehensive 2x2 summary figure."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Learning Curves (top left)
    ax1 = fig.add_subplot(gs[0, :])
    
    baseline_methods = [m for m in data.keys() if METHODS.get(m, {}).get('category') == 'baseline']
    adaptive_methods = [m for m in data.keys() if METHODS.get(m, {}).get('category') == 'adaptive']
    adaptive_methods.sort(key=lambda m: data[m]['final_score'], reverse=True)
    methods_ordered = baseline_methods + adaptive_methods
    
    for method in methods_ordered:
        if method not in METHODS:
            continue
        config = METHODS[method]
        method_data = data[method]
        scores_smooth = smooth_curve(method_data['scores'], sigma=1.5)
        
        ax1.plot(method_data['steps'], scores_smooth,
                label=config['name'], color=config['color'],
                linestyle=config['linestyle'], linewidth=config['linewidth'],
                alpha=config['alpha'], zorder=config['zorder'])
    
    ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Learning Curves', fontsize=14, fontweight='bold', loc='left')
    ax1.legend(loc='lower right', ncol=2, fontsize=9, frameon=True)
    ax1.grid(True, alpha=0.25)
    ax1.set_xlim(0, 50000)
    
    # 2. Final Performance (bottom left)
    ax2 = fig.add_subplot(gs[1, 0])
    
    methods_sorted = sorted(data.items(), key=lambda x: x[1]['final_score'], reverse=True)
    names = [METHODS[m]['short'] for m, _ in methods_sorted if m in METHODS]
    scores = [d['final_score'] for m, d in methods_sorted if m in METHODS]
    colors = [METHODS[m]['color'] for m, _ in methods_sorted if m in METHODS]
    
    bars = ax2.bar(range(len(names)), scores, color=colors, alpha=0.85,
                   edgecolor='black', linewidth=1.5)
    
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax2.text(i, score + 0.5, f'{score:.1f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Final Score', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Final Performance', fontsize=14, fontweight='bold', loc='left')
    ax2.grid(True, axis='y', alpha=0.25)
    ax2.set_ylim(0, max(scores) * 1.15)
    
    # 3. Improvement (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    
    baseline_score = data['baseline_no_shaping']['final_score']
    improvements = [((s - baseline_score) / baseline_score * 100) for s in scores[1:]]
    imp_names = names[1:]
    imp_colors = [colors[i+1] for i in range(len(improvements))]
    
    bars = ax3.barh(range(len(imp_names)), improvements, color=imp_colors, alpha=0.85,
                    edgecolor='black', linewidth=1.5)
    
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        label_x = imp + 1 if imp > 0 else imp - 1
        ha = 'left' if imp > 0 else 'right'
        ax3.text(label_x, i, f'{imp:+.1f}%', va='center', ha=ha,
                fontsize=9, fontweight='bold')
    
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8)
    ax3.set_yticks(range(len(imp_names)))
    ax3.set_yticklabels(imp_names, fontsize=9)
    ax3.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax3.set_title('(C) Improvement over Baseline', fontsize=14, fontweight='bold', loc='left')
    ax3.grid(True, axis='x', alpha=0.25)
    
    # Overall title
    fig.suptitle('Adaptive Reward Shaping: Comprehensive Results Summary', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(f"{OUTPUT_DIR}/comprehensive_summary.png", bbox_inches='tight', 
                facecolor='white', dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/comprehensive_summary.pdf", bbox_inches='tight', 
                facecolor='white')
    print(f"Created: comprehensive_summary.png/pdf")
    plt.close()


def main():
    print("\n" + "="*80)
    print("CREATING PUBLICATION-QUALITY FIGURES")
    print("="*80 + "\n")
    
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} experiments\n")
    
    print("Generating figures...")
    print("-"*80)
    
    create_learning_curves(data)
    create_final_performance_comparison(data)
    create_improvement_heatmap(data)
    create_category_performance(data)
    create_summary_figure(data)
    
    print("-"*80)
    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
    print("Generated both PNG and PDF versions for publication")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
