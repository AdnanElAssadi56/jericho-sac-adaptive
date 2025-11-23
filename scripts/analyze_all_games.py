#!/usr/bin/env python3
"""
Analyze results across ALL games (zork1, detective, pentari, adventureland, jewel, zork3)
Generate comprehensive statistics for paper
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import pickle

GAMES = ["zork1", "detective", "pentari", "adventureland", "jewel", "zork3"]
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

def load_progress(game, seed, method):
    """Load progress.json and extract learning curve"""
    path = Path(f"logging/final/{game}/{seed}/{method}/progress.json")
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
                steps.append((i + 1) * 100)
            except:
                continue
    
    if not scores:
        return None
    
    return {'steps': steps, 'scores': scores}

def analyze_game(game):
    """Analyze results for a single game"""
    print(f"\n{'='*80}")
    print(f"ANALYZING {game.upper()}")
    print(f"{'='*80}\n")
    
    # Collect all data
    all_data = defaultdict(lambda: {'seeds': {}, 'final_scores': []})
    
    for method in METHODS:
        for seed in SEEDS:
            data = load_progress(game, seed, method)
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
        std_score = np.std(scores, ddof=1) if len(scores) > 1 else 0
        
        results.append({
            'method': method,
            'name': METHOD_NAMES[method],
            'mean': mean_score,
            'std': std_score,
            'scores': scores,
            'n_seeds': len(scores),
            'seeds': all_data[method]['seeds']
        })
    
    # Sort by mean score
    results.sort(key=lambda x: x['mean'], reverse=True)
    
    # Print results
    print(f"{'Rank':<6} {'Method':<30} {'Mean':<12} {'Std':<10} {'N':<5}")
    print("-"*80)
    
    baseline_static = next((r for r in results if r['method'] == 'baseline_static_shaping'), None)
    baseline_static_mean = baseline_static['mean'] if baseline_static else 0
    
    for i, r in enumerate(results, 1):
        improvement = ((r['mean'] - baseline_static_mean) / baseline_static_mean * 100) if baseline_static_mean > 0 else 0
        print(f"{i:<6} {r['name']:<30} {r['mean']:<12.2f} {r['std']:<10.2f} {r['n_seeds']:<5}")
    
    return results, baseline_static

def main():
    print("="*80)
    print("MULTI-GAME ANALYSIS - ALL RESULTS")
    print("="*80)
    
    all_game_results = {}
    
    # Analyze each game
    for game in GAMES:
        results, baseline = analyze_game(game)
        all_game_results[game] = {
            'results': results,
            'baseline_static': baseline
        }
    
    # Save all results
    Path('results/data').mkdir(parents=True, exist_ok=True)
    with open('results/data/all_games_results.pkl', 'wb') as f:
        pickle.dump(all_game_results, f)
    
    print(f"\n{'='*80}")
    print("CROSS-GAME SUMMARY")
    print(f"{'='*80}\n")
    
    # Aggregate statistics across games
    method_wins = defaultdict(int)
    method_avg_improvement = defaultdict(list)
    
    for game, data in all_game_results.items():
        results = data['results']
        baseline = data['baseline_static']
        
        if results and baseline:
            # Count wins
            best = results[0]
            method_wins[best['method']] += 1
            
            # Track improvements
            for r in results:
                if 'adaptive' in r['method']:
                    improvement = ((r['mean'] - baseline['mean']) / baseline['mean'] * 100)
                    method_avg_improvement[r['method']].append(improvement)
    
    print("Method Performance Across Games:")
    print("-"*80)
    print(f"{'Method':<30} {'Wins':<10} {'Avg Improvement':<20}")
    print("-"*80)
    
    for method in METHODS:
        if 'adaptive' in method:
            wins = method_wins.get(method, 0)
            improvements = method_avg_improvement.get(method, [])
            avg_imp = np.mean(improvements) if improvements else 0
            std_imp = np.std(improvements) if len(improvements) > 1 else 0
            print(f"{METHOD_NAMES[method]:<30} {wins:<10} {avg_imp:>6.1f}% ± {std_imp:.1f}%")
    
    print(f"\nResults saved to results/data/all_games_results.pkl")
    
    # Generate LaTeX table for multi-game results
    print(f"\n{'='*80}")
    print("LATEX TABLE - MULTI-GAME RESULTS")
    print(f"{'='*80}\n")
    
    print("\\begin{table*}[t]")
    print("\\caption{Performance across multiple games. Mean $\\pm$ std over available seeds.}")
    print("\\label{tab:multi_game}")
    print("\\vskip 0.15in")
    print("\\begin{center}")
    print("\\begin{small}")
    print("\\begin{sc}")
    print("\\begin{tabular}{l" + "c"*len(GAMES) + "}")
    print("\\toprule")
    print("Method & " + " & ".join([g.capitalize() for g in GAMES]) + " \\\\")
    print("\\midrule")
    
    # Print each method's results across games
    for method in METHODS:
        row = [METHOD_NAMES[method]]
        for game in GAMES:
            game_data = all_game_results[game]
            method_result = next((r for r in game_data['results'] if r['method'] == method), None)
            if method_result and method_result['n_seeds'] > 0:
                row.append(f"{method_result['mean']:.1f} $\\pm$ {method_result['std']:.1f}")
            else:
                row.append("--")
        print(" & ".join(row) + " \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{sc}")
    print("\\end{small}")
    print("\\end{center}")
    print("\\vskip -0.1in")
    print("\\end{table*}")

if __name__ == '__main__':
    main()
