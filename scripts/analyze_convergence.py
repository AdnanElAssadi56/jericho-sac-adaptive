#!/usr/bin/env python3
"""
Analyze convergence properties of different schedulers
- Sample efficiency: Steps to reach threshold performance
- Learning speed: Slope of learning curve
- Stability: Variance over time
- Final convergence: Plateau detection
"""
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats

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

def load_learning_curve(game, seed, method):
    """Load full learning curve"""
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

def compute_convergence_metrics(curve, threshold_percentile=0.8):
    """
    Compute convergence metrics for a learning curve
    
    Returns:
    - steps_to_threshold: Steps to reach 80% of final performance
    - learning_rate: Average improvement per 1000 steps (early phase)
    - stability: Coefficient of variation in final 25% of training
    - converged: Whether curve has plateaued
    """
    if curve is None or len(curve) < 100:
        return None
    
    final_score = curve[-1]
    threshold = threshold_percentile * final_score
    
    # Steps to threshold
    steps_to_threshold = None
    for i, score in enumerate(curve):
        if score >= threshold:
            steps_to_threshold = (i + 1) * 100  # Convert to actual steps
            break
    
    # Learning rate (first 25% of training)
    early_phase = curve[:len(curve)//4]
    if len(early_phase) > 10:
        # Fit linear regression
        x = np.arange(len(early_phase))
        slope, _, _, _, _ = stats.linregress(x, early_phase)
        learning_rate = slope * 10  # Per 1000 steps
    else:
        learning_rate = 0
    
    # Stability (final 25% of training)
    late_phase = curve[-len(curve)//4:]
    if len(late_phase) > 0 and np.mean(late_phase) > 0:
        stability = np.std(late_phase) / np.mean(late_phase)  # Coefficient of variation
    else:
        stability = float('inf')
    
    # Convergence detection (is final 10% flat?)
    final_10pct = curve[-len(curve)//10:]
    if len(final_10pct) > 5:
        # Check if slope is near zero
        x = np.arange(len(final_10pct))
        slope, _, _, p_value, _ = stats.linregress(x, final_10pct)
        converged = (p_value > 0.05)  # Not significantly different from flat
    else:
        converged = False
    
    return {
        'steps_to_threshold': steps_to_threshold,
        'learning_rate': learning_rate,
        'stability': stability,
        'converged': converged,
        'final_score': final_score
    }

def analyze_game_convergence(game):
    """Analyze convergence for all methods on a game"""
    print(f"\n{'='*80}")
    print(f"CONVERGENCE ANALYSIS: {game.upper()}")
    print(f"{'='*80}\n")
    
    results = {}
    
    for method in METHODS:
        curves = []
        for seed in SEEDS:
            curve = load_learning_curve(game, seed, method)
            if curve is not None:
                curves.append(curve)
        
        if not curves:
            continue
        
        # Compute metrics for each seed
        metrics_per_seed = [compute_convergence_metrics(c) for c in curves]
        metrics_per_seed = [m for m in metrics_per_seed if m is not None]
        
        if not metrics_per_seed:
            continue
        
        # Aggregate across seeds
        results[method] = {
            'steps_to_threshold': np.mean([m['steps_to_threshold'] for m in metrics_per_seed if m['steps_to_threshold']]),
            'learning_rate': np.mean([m['learning_rate'] for m in metrics_per_seed]),
            'stability': np.mean([m['stability'] for m in metrics_per_seed]),
            'converged_count': sum([m['converged'] for m in metrics_per_seed]),
            'final_score': np.mean([m['final_score'] for m in metrics_per_seed]),
            'n_seeds': len(metrics_per_seed)
        }
    
    # Print results
    print(f"{'Method':<30} {'Steps to 80%':<15} {'Learn Rate':<15} {'Stability':<12} {'Converged'}")
    print("-"*90)
    
    # Sort by final score
    sorted_methods = sorted(results.items(), key=lambda x: x[1]['final_score'], reverse=True)
    
    for method, metrics in sorted_methods:
        steps = metrics['steps_to_threshold']
        rate = metrics['learning_rate']
        stab = metrics['stability']
        conv = f"{metrics['converged_count']}/{metrics['n_seeds']}"
        
        steps_str = f"{int(steps):,}" if not np.isnan(steps) else "N/A"
        rate_str = f"{rate:.2f}" if not np.isnan(rate) else "N/A"
        stab_str = f"{stab:.3f}" if not np.isnan(stab) and stab != float('inf') else "N/A"
        
        print(f"{METHOD_NAMES[method]:<30} {steps_str:<15} {rate_str:<15} {stab_str:<12} {conv}")
    
    return results

def main():
    print("="*80)
    print("CONVERGENCE ANALYSIS - MULTI-GAME")
    print("="*80)
    print()
    print("Metrics:")
    print("  - Steps to 80%: Steps needed to reach 80% of final performance")
    print("  - Learn Rate: Average score improvement per 1000 steps (early training)")
    print("  - Stability: Coefficient of variation in final 25% (lower = more stable)")
    print("  - Converged: Number of seeds that plateaued in final 10%")
    print()
    
    all_results = {}
    
    for game in GAMES:
        results = analyze_game_convergence(game)
        all_results[game] = results
    
    # Cross-game summary
    print(f"\n{'='*80}")
    print("CROSS-GAME CONVERGENCE SUMMARY")
    print(f"{'='*80}\n")
    
    print("Sample Efficiency (Average steps to 80% across games):")
    print("-"*80)
    
    method_avg_steps = {}
    for method in METHODS:
        steps_list = []
        for game in GAMES:
            if game in all_results and method in all_results[game]:
                steps = all_results[game][method]['steps_to_threshold']
                if not np.isnan(steps):
                    steps_list.append(steps)
        
        if steps_list:
            method_avg_steps[method] = np.mean(steps_list)
    
    # Sort by sample efficiency (lower is better)
    sorted_by_efficiency = sorted(method_avg_steps.items(), key=lambda x: x[1])
    
    for method, avg_steps in sorted_by_efficiency:
        print(f"  {METHOD_NAMES[method]:<30} {int(avg_steps):,} steps")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80 + "\n")
    
    # Find most sample efficient
    if sorted_by_efficiency:
        best_method, best_steps = sorted_by_efficiency[0]
        print(f"Most Sample Efficient: {METHOD_NAMES[best_method]} ({int(best_steps):,} steps)")
    
    # Find most stable
    method_avg_stability = {}
    for method in METHODS:
        stab_list = []
        for game in GAMES:
            if game in all_results and method in all_results[game]:
                stab = all_results[game][method]['stability']
                if not np.isnan(stab) and stab != float('inf'):
                    stab_list.append(stab)
        
        if stab_list:
            method_avg_stability[method] = np.mean(stab_list)
    
    if method_avg_stability:
        most_stable = min(method_avg_stability.items(), key=lambda x: x[1])
        print(f"Most Stable: {METHOD_NAMES[most_stable[0]]} (CV = {most_stable[1]:.3f})")
    
    # Convergence rates
    print("\nConvergence Rates:")
    for method in METHODS:
        converged_total = 0
        total_seeds = 0
        for game in GAMES:
            if game in all_results and method in all_results[game]:
                converged_total += all_results[game][method]['converged_count']
                total_seeds += all_results[game][method]['n_seeds']
        
        if total_seeds > 0:
            conv_rate = (converged_total / total_seeds) * 100
            print(f"  {METHOD_NAMES[method]:<30} {conv_rate:.0f}% ({converged_total}/{total_seeds})")

if __name__ == '__main__':
    main()
