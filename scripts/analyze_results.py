#!/usr/bin/env python3
"""
Analyze Zork1 results across all 3 seeds (0, 1, 3)
Generate accurate statistics and data for paper
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats

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
    'adaptive_time_decay_exp': 'Exponential Decay',
    'adaptive_time_decay_linear': 'Linear Decay',
    'adaptive_time_decay_cosine': 'Cosine Decay',
    'adaptive_sparsity_triggered': 'Sparsity-Triggered',
    'adaptive_sparsity_sensitive': 'Sparsity-Sensitive',
    'adaptive_uncertainty_informed': 'Entropy-Informed'
}

def load_progress(seed, method):
    """Load progress.json and extract learning curve"""
    path = Path(f"logging/final/{GAME}/{seed}/{method}/progress.json")
    if not path.exists():
        return None
    
    scores = []
    steps = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                score = data.get('train/Last100EpisodeScores', 0)
                scores.append(score)
                steps.append((i + 1) * 100)  # Each entry is 100 steps
            except:
                continue
    
    return {'steps': steps, 'scores': scores}

def main():
    print("="*80)
    print(f"ANALYZING ZORK1 RESULTS - 3 SEEDS (0, 1, 3)")
    print("="*80)
    print()
    
    # Collect all data
    all_data = defaultdict(lambda: {'seeds': {}, 'final_scores': []})
    
    for method in METHODS:
        for seed in SEEDS:
            data = load_progress(seed, method)
            if data:
                all_data[method]['seeds'][seed] = data
                all_data[method]['final_scores'].append(data['scores'][-1])
    
    # Calculate statistics
    results = []
    for method in METHODS:
        if not all_data[method]['final_scores']:
            continue
        
        scores = all_data[method]['final_scores']
        mean_score = np.mean(scores)
        std_score = np.std(scores, ddof=1)  # Sample std
        
        results.append({
            'method': method,
            'name': METHOD_NAMES[method],
            'mean': mean_score,
            'std': std_score,
            'scores': scores,
            'seeds': all_data[method]['seeds']
        })
    
    # Sort by mean score
    results.sort(key=lambda x: x['mean'], reverse=True)
    
    # Print main results table
    print("MAIN RESULTS TABLE (Mean ± Std over 3 seeds)")
    print("-"*80)
    print(f"{'Rank':<6} {'Method':<30} {'Mean':<12} {'Std':<10} {'Seeds'}")
    print("-"*80)
    
    baseline_static = next((r for r in results if r['method'] == 'baseline_static_shaping'), None)
    baseline_static_mean = baseline_static['mean'] if baseline_static else 0
    
    for i, r in enumerate(results, 1):
        improvement = ((r['mean'] - baseline_static_mean) / baseline_static_mean * 100) if baseline_static_mean > 0 else 0
        seed_str = ", ".join([f"{s:.2f}" for s in r['scores']])
        print(f"{i:<6} {r['name']:<30} {r['mean']:<12.2f} {r['std']:<10.2f} [{seed_str}]")
    
    print()
    print("="*80)
    print("LATEX TABLE FORMAT")
    print("="*80)
    print()
    
    # Generate LaTeX table
    print("\\begin{table}[t]")
    print("\\caption{Final performance on Zork1 (50k steps). Mean $\\pm$ std over 3 seeds.}")
    print("\\label{tab:main_results}")
    print("\\vskip 0.15in")
    print("\\begin{center}")
    print("\\begin{small}")
    print("\\begin{sc}")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Method & Score & vs Static \\\\")
    print("\\midrule")
    
    # Baselines
    print("\\multicolumn{3}{c}{\\textit{Baselines}} \\\\")
    for r in results:
        if 'baseline' in r['method']:
            improvement = ((r['mean'] - baseline_static_mean) / baseline_static_mean * 100) if baseline_static_mean > 0 else 0
            if r['method'] == 'baseline_static_shaping':
                print(f"{r['name']} & {r['mean']:.2f} $\\pm$ {r['std']:.2f} & -- \\\\")
            else:
                print(f"{r['name']} & {r['mean']:.2f} $\\pm$ {r['std']:.2f} & {improvement:+.1f}\\% \\\\")
    
    print("\\midrule")
    print("\\multicolumn{3}{c}{\\textit{Time-Based Adaptive}} \\\\")
    for r in results:
        if 'time_decay' in r['method']:
            improvement = ((r['mean'] - baseline_static_mean) / baseline_static_mean * 100)
            bold_start = "\\textbf{" if r == results[0] else ""
            bold_end = "}" if r == results[0] else ""
            print(f"{bold_start}{r['name']}{bold_end} & {bold_start}{r['mean']:.2f} $\\pm$ {r['std']:.2f}{bold_end} & {bold_start}{improvement:+.1f}\\%{bold_end} \\\\")
    
    print("\\midrule")
    print("\\multicolumn{3}{c}{\\textit{Sparsity-Based Adaptive}} \\\\")
    for r in results:
        if 'sparsity' in r['method']:
            improvement = ((r['mean'] - baseline_static_mean) / baseline_static_mean * 100)
            bold_start = "\\textbf{" if r == results[0] else ""
            bold_end = "}" if r == results[0] else ""
            print(f"{bold_start}{r['name']}{bold_end} & {bold_start}{r['mean']:.2f} $\\pm$ {r['std']:.2f}{bold_end} & {bold_start}{improvement:+.1f}\\%{bold_end} \\\\")
    
    print("\\midrule")
    print("\\multicolumn{3}{c}{\\textit{Uncertainty-Based Adaptive}} \\\\")
    for r in results:
        if 'uncertainty' in r['method']:
            improvement = ((r['mean'] - baseline_static_mean) / baseline_static_mean * 100)
            print(f"{r['name']} & {r['mean']:.2f} $\\pm$ {r['std']:.2f} & {improvement:+.1f}\\% \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{sc}")
    print("\\end{small}")
    print("\\end{center}")
    print("\\vskip -0.1in")
    print("\\end{table}")
    
    print()
    print("="*80)
    print("PER-SEED TABLE")
    print("="*80)
    print()
    
    print("\\begin{table}[t]")
    print("\\caption{Per-seed results on Zork1 (50k steps).}")
    print("\\label{tab:per_seed}")
    print("\\vskip 0.15in")
    print("\\begin{center}")
    print("\\begin{small}")
    print("\\begin{sc}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("Method & Seed 0 & Seed 1 & Seed 3 \\\\")
    print("\\midrule")
    
    for r in results:
        seed_scores = [r['seeds'][s]['scores'][-1] if s in r['seeds'] else 0 for s in SEEDS]
        print(f"{r['name']} & {seed_scores[0]:.2f} & {seed_scores[1]:.2f} & {seed_scores[2]:.2f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{sc}")
    print("\\end{small}")
    print("\\end{center}")
    print("\\vskip -0.1in")
    print("\\end{table}")
    
    print()
    print("="*80)
    print("KEY STATISTICS")
    print("="*80)
    print()
    
    best = results[0]
    print(f"Best Method: {best['name']}")
    print(f"   Mean: {best['mean']:.2f} ± {best['std']:.2f}")
    print(f"   Seeds: {best['scores']}")
    print()
    
    if baseline_static:
        print(f"Static Shaping Baseline: {baseline_static['mean']:.2f} ± {baseline_static['std']:.2f}")
        improvement = ((best['mean'] - baseline_static['mean']) / baseline_static['mean'] * 100)
        print(f"   Best vs Static: {improvement:+.1f}%")
        print()
    
    baseline_no = next((r for r in results if r['method'] == 'baseline_no_shaping'), None)
    if baseline_no:
        print(f"No Shaping Baseline: {baseline_no['mean']:.2f} ± {baseline_no['std']:.2f}")
        improvement = ((best['mean'] - baseline_no['mean']) / baseline_no['mean'] * 100)
        print(f"   Best vs No Shaping: {improvement:+.1f}%")
        print()
    
    # Count adaptive methods beating static
    if baseline_static:
        adaptive_better = [r for r in results if 'adaptive' in r['method'] and r['mean'] > baseline_static['mean']]
        print(f"Adaptive methods beating static: {len(adaptive_better)}/6")
        for r in adaptive_better:
            improvement = ((r['mean'] - baseline_static['mean']) / baseline_static['mean'] * 100)
            print(f"   - {r['name']}: {r['mean']:.2f} ({improvement:+.1f}%)")
    
    print()
    print("="*80)
    print("STATISTICAL SIGNIFICANCE TESTS (Welch's t-test)")
    print("="*80)
    print()
    
    if baseline_static:
        print("Key comparisons vs Static Shaping:")
        print("-" * 60)
        
        # Test each adaptive method vs static
        for r in results:
            if 'adaptive' in r['method'] and len(r['scores']) == 3:
                # Welch's t-test (doesn't assume equal variances)
                t_stat, p_value = stats.ttest_ind(r['scores'], baseline_static['scores'], equal_var=False)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((r['std']**2 + baseline_static['std']**2) / 2)
                cohens_d = (r['mean'] - baseline_static['mean']) / pooled_std if pooled_std > 0 else 0
                
                sig_marker = ""
                if p_value < 0.001:
                    sig_marker = "***"
                elif p_value < 0.01:
                    sig_marker = "**"
                elif p_value < 0.05:
                    sig_marker = "*"
                
                improvement = ((r['mean'] - baseline_static['mean']) / baseline_static['mean'] * 100)
                
                print(f"{r['name']:<30} p={p_value:.4f} {sig_marker:<4} d={cohens_d:+.2f}  ({improvement:+.1f}%)")
        
        print()
        print("Significance levels: * p<0.05, ** p<0.01, *** p<0.001")
        print()
        
        # Best method vs static
        if len(best['scores']) == 3 and len(baseline_static['scores']) == 3:
            t_stat, p_value = stats.ttest_ind(best['scores'], baseline_static['scores'], equal_var=False)
            print(f"Best method ({best['name']}) vs Static Shaping:")
            print(f"   t-statistic: {t_stat:.3f}")
            print(f"   p-value: {p_value:.4f}")
            if p_value < 0.05:
                print(f"   Statistically significant at alpha=0.05")
            else:
                print(f"   Not statistically significant at alpha=0.05")
            print()
        
        # No shaping vs static (interesting comparison)
        if baseline_no and len(baseline_no['scores']) == 3:
            t_stat, p_value = stats.ttest_ind(baseline_no['scores'], baseline_static['scores'], equal_var=False)
            print(f"No Shaping vs Static Shaping:")
            print(f"   t-statistic: {t_stat:.3f}")
            print(f"   p-value: {p_value:.4f}")
            if p_value < 0.05:
                print(f"   Statistically significant at alpha=0.05")
            else:
                print(f"   Not statistically significant at alpha=0.05")
            print()
    
    print()
    print("="*80)
    print("COMPARISON TO LITERATURE (Li et al. 2023)")
    print("="*80)
    print()
    
    if baseline_no:
        print(f"No Shaping:")
        print(f"  Paper: 25.7")
        print(f"  Ours:  {baseline_no['mean']:.2f} ± {baseline_no['std']:.2f}")
        print()
    
    if baseline_static:
        print(f"Static Shaping:")
        print(f"  Paper: 36.0")
        print(f"  Ours:  {baseline_static['mean']:.2f} ± {baseline_static['std']:.2f}")
        print()
    
    print(f"Best Adaptive:")
    print(f"  Paper: N/A")
    print(f"  Ours:  {best['mean']:.2f} ± {best['std']:.2f}")
    
    # Save data for plotting
    import pickle
    Path('results/data').mkdir(parents=True, exist_ok=True)
    with open('results/data/zork1_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print()
    print("Data saved to results/data/zork1_results.pkl")

if __name__ == '__main__':
    main()
