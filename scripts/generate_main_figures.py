#!/usr/bin/env python3
"""
Generate essential figures for main paper
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# Publication settings
plt.rcParams['figure.figsize'] = (7, 4.5)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

GAMES = ["zork1", "detective", "pentari"]
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

def load_learning_curve(game, seed, method):
    path = Path(f"logging/final/{game}/{seed}/{method}/progress.json")
    if not path.exists():
        return None
    
    scores = []
    with open(path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                score = data.get('train/Last100EpisodeScores', 0)
                scores.append(score)
            except:
                continue
    
    return np.array(scores) if scores else None

def plot_training_progress():
    """Figure 1: Training progress across 3 games (3 subplots)"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for game_idx, game in enumerate(GAMES):
        ax = axes[game_idx]
        
        # Plot all methods
        for method in METHODS:
            curves = []
            for seed in SEEDS:
                curve = load_learning_curve(game, seed, method)
                if curve is not None:
                    curves.append(curve)
            
            if not curves:
                continue
            
            # Pad to same length
            max_len = max(len(c) for c in curves)
            padded = [np.pad(c, (0, max_len - len(c)), mode='edge') if len(c) < max_len else c 
                     for c in curves]
            
            curves_array = np.array(padded)
            mean_curve = np.mean(curves_array, axis=0)
            std_curve = np.std(curves_array, axis=0)
            steps = np.arange(len(mean_curve)) * 100
            
            ax.plot(steps, mean_curve, label=METHOD_NAMES[method], 
                   color=COLORS[method], linewidth=2, alpha=0.9)
            ax.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve,
                           color=COLORS[method], alpha=0.15)
        
        ax.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
        if game_idx == 0:
            ax.set_ylabel('Average Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{game.capitalize()}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 50000)
        
        if game_idx == 2:
            ax.legend(loc='best', framealpha=0.95, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/figures/training_progress.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('results/figures/training_progress.png', bbox_inches='tight', dpi=300)
    print("Saved training_progress.pdf/png")
    plt.close()

def plot_performance_heatmap():
    """Figure 2: Heatmap showing which methods work where"""
    # Load results
    with open('results/data/multi_game_results.pkl', 'rb') as f:
        all_results = pickle.load(f)
    
    # Create improvement matrix
    improvements = []
    method_labels = []
    
    for method in METHODS:
        if method.startswith('baseline'):
            continue
        row = []
        for game in GAMES:
            if game in all_results:
                baseline = all_results[game]['baseline_static']
                method_result = next((r for r in all_results[game]['results'] 
                                    if r['method'] == method), None)
                if method_result and baseline:
                    improvement = ((method_result['mean'] - baseline['mean']) / 
                                 baseline['mean'] * 100)
                    row.append(improvement)
                else:
                    row.append(0)
            else:
                row.append(0)
        improvements.append(row)
        method_labels.append(METHOD_NAMES[method])
    
    improvements = np.array(improvements)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(improvements, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=60)
    
    # Set ticks
    ax.set_xticks(np.arange(len(GAMES)))
    ax.set_yticks(np.arange(len(method_labels)))
    ax.set_xticklabels([g.capitalize() for g in GAMES], fontsize=11, fontweight='bold')
    ax.set_yticklabels(method_labels, fontsize=10)
    
    # Add text annotations
    for i in range(len(method_labels)):
        for j in range(len(GAMES)):
            text = ax.text(j, i, f'{improvements[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=9,
                          fontweight='bold')
    
    ax.set_title('Improvement over Static Shaping (%)', 
                fontsize=13, fontweight='bold', pad=15)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Improvement (%)', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    plt.savefig('results/figures/performance_heatmap.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('results/figures/performance_heatmap.png', bbox_inches='tight', dpi=300)
    print("Saved performance_heatmap.pdf/png")
    plt.close()

if __name__ == '__main__':
    print("Generating main paper figures...")
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    plot_training_progress()
    plot_performance_heatmap()
    print("\nAll main figures generated!")
