#!/usr/bin/env python3
"""
Generate all figures for the paper with proper formatting
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# Set publication-quality defaults
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.dpi'] = 300

GAME = "zork1"
SEEDS = ["seed0", "seed1", "seed3"]
METHODS = [
    'baseline_no_shaping',
    'baseline_static_shaping',
    'adaptive_time_decay_exp',
    'adaptive_time_decay_linear',
    'adaptive_time_decay_cosine',
    'adaptive_sparsity_triggered',
    'adaptive_sparsity_sensitive',
    'adaptive_uncertainty_informed'
]

METHOD_NAMES = {
    'baseline_no_shaping': 'No Shaping',
    'baseline_static_shaping': 'Static Shaping',
    'adaptive_time_decay_exp': 'Exponential',
    'adaptive_time_decay_linear': 'Linear',
    'adaptive_time_decay_cosine': 'Cosine',
    'adaptive_sparsity_triggered': 'Sparsity-Triggered',
    'adaptive_sparsity_sensitive': 'Sparsity-Sensitive',
    'adaptive_uncertainty_informed': 'Entropy-Informed'
}

COLORS = {
    'baseline_no_shaping': '#999999',
    'baseline_static_shaping': '#666666',
    'adaptive_time_decay_exp': '#1f77b4',
    'adaptive_time_decay_linear': '#ff7f0e',
    'adaptive_time_decay_cosine': '#2ca02c',
    'adaptive_sparsity_triggered': '#d62728',
    'adaptive_sparsity_sensitive': '#9467bd',
    'adaptive_uncertainty_informed': '#8c564b'
}

def load_learning_curves(method):
    """Load learning curves for all seeds"""
    curves = []
    for seed in SEEDS:
        path = Path(f"logging/final/{GAME}/{seed}/{method}/progress.json")
        if not path.exists():
            continue
        
        scores = []
        with open(path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    score = data.get('train/Last100EpisodeScores', 0)
                    scores.append(score)
                except:
                    continue
        
        if scores:
            curves.append(np.array(scores))
    
    return curves

def plot_learning_curves():
    """Figure 1: Learning curves for all methods"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Plot each method
    for method in METHODS:
        curves = load_learning_curves(method)
        if not curves:
            continue
        
        # Pad curves to same length
        max_len = max(len(c) for c in curves)
        padded = []
        for c in curves:
            if len(c) < max_len:
                padded.append(np.pad(c, (0, max_len - len(c)), mode='edge'))
            else:
                padded.append(c)
        
        curves_array = np.array(padded)
        mean_curve = np.mean(curves_array, axis=0)
        std_curve = np.std(curves_array, axis=0)
        steps = np.arange(len(mean_curve)) * 100
        
        # Plot mean line
        ax.plot(steps, mean_curve, label=METHOD_NAMES[method], 
                color=COLORS[method], linewidth=1.5, alpha=0.9)
        
        # Plot std as shaded region
        ax.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve,
                        color=COLORS[method], alpha=0.15)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Score (Last 100 Episodes)')
    ax.set_title('Learning Curves on Zork1')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50000)
    
    plt.tight_layout()
    plt.savefig('results/figures/learning_curves.pdf', bbox_inches='tight')
    plt.savefig('results/figures/learning_curves.png', bbox_inches='tight', dpi=300)
    print("Saved learning_curves.pdf/png")
    plt.close()

def plot_scheduler_behavior():
    """Figure 2: Scheduler behavior over time"""
    fig, ax = plt.subplots(figsize=(7, 4))
    
    steps = np.arange(0, 50001, 100)
    alpha_0 = 1.0
    
    # Exponential decay
    lambda_exp = 2e-5
    alpha_exp = alpha_0 * np.exp(-lambda_exp * steps)
    ax.plot(steps, alpha_exp, label='Exponential', color=COLORS['adaptive_time_decay_exp'], linewidth=2)
    
    # Linear decay
    alpha_lin = alpha_0 * (1 - steps / 50000)
    ax.plot(steps, alpha_lin, label='Linear', color=COLORS['adaptive_time_decay_linear'], linewidth=2)
    
    # Cosine decay
    alpha_cos = alpha_0 * (1 + np.cos(np.pi * steps / 50000)) / 2
    ax.plot(steps, alpha_cos, label='Cosine', color=COLORS['adaptive_time_decay_cosine'], linewidth=2)
    
    # Static
    alpha_static = np.ones_like(steps) * alpha_0
    ax.plot(steps, alpha_static, label='Static', color=COLORS['baseline_static_shaping'], 
            linewidth=2, linestyle='--')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Shaping Coefficient α(t)')
    ax.set_title('Time-Based Scheduler Behavior')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50000)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('results/figures/scheduler_behavior.pdf', bbox_inches='tight')
    plt.savefig('results/figures/scheduler_behavior.png', bbox_inches='tight', dpi=300)
    print("Saved scheduler_behavior.pdf/png")
    plt.close()

def plot_final_comparison():
    """Figure 3: Bar chart of final performance"""
    # Load results
    with open('results/data/zork1_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    methods = [r['name'] for r in results]
    means = [r['mean'] for r in results]
    stds = [r['std'] for r in results]
    colors_list = [COLORS[r['method']] for r in results]
    
    y_pos = np.arange(len(methods))
    
    bars = ax.barh(y_pos, means, xerr=stds, color=colors_list, alpha=0.8, 
                   capsize=5, error_kw={'linewidth': 1.5})
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel('Score (Mean ± Std over 3 seeds)')
    ax.set_title('Final Performance on Zork1 (50k steps)')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(mean + std + 1, i, f'{mean:.1f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/figures/final_comparison.pdf', bbox_inches='tight')
    plt.savefig('results/figures/final_comparison.png', bbox_inches='tight', dpi=300)
    print("Saved final_comparison.pdf/png")
    plt.close()

def plot_improvement_over_static():
    """Figure 4: Improvement over static shaping"""
    with open('results/data/zork1_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    baseline_static = next(r for r in results if r['method'] == 'baseline_static_shaping')
    baseline_mean = baseline_static['mean']
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    adaptive_results = [r for r in results if 'adaptive' in r['method']]
    methods = [r['name'] for r in adaptive_results]
    improvements = [((r['mean'] - baseline_mean) / baseline_mean * 100) for r in adaptive_results]
    colors_list = [COLORS[r['method']] for r in adaptive_results]
    
    # Sort by improvement
    sorted_data = sorted(zip(methods, improvements, colors_list), key=lambda x: x[1], reverse=True)
    methods, improvements, colors_list = zip(*sorted_data)
    
    y_pos = np.arange(len(methods))
    
    bars = ax.barh(y_pos, improvements, color=colors_list, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel('Improvement over Static Shaping (%)')
    ax.set_title('Adaptive Methods vs Static Shaping Baseline')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for i, imp in enumerate(improvements):
        ax.text(imp + 1, i, f'{imp:+.1f}%', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/figures/improvement_comparison.pdf', bbox_inches='tight')
    plt.savefig('results/figures/improvement_comparison.png', bbox_inches='tight', dpi=300)
    print("Saved improvement_comparison.pdf/png")
    plt.close()

def main():
    print("="*80)
    print("GENERATING PAPER FIGURES")
    print("="*80)
    print()
    
    # Create output directory
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    
    print("Generating Figure 1: Learning curves...")
    plot_learning_curves()
    
    print("Generating Figure 2: Scheduler behavior...")
    plot_scheduler_behavior()
    
    print("Generating Figure 3: Final comparison...")
    plot_final_comparison()
    
    print("Generating Figure 4: Improvement comparison...")
    plot_improvement_over_static()
    
    print()
    print("="*80)
    print("ALL FIGURES GENERATED")
    print("="*80)
    print()
    print("Figures saved to results/figures/:")
    print("  - learning_curves.pdf/png")
    print("  - scheduler_behavior.pdf/png")
    print("  - final_comparison.pdf/png")
    print("  - improvement_comparison.pdf/png")

if __name__ == '__main__':
    main()
