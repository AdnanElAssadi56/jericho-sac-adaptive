#!/usr/bin/env python3
"""
Analysis script for adaptive reward shaping experimental results.
Author: Adnan El Assadi
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
from pathlib import Path
import glob


def load_tensorboard_data(log_dir):
    """Load data from TensorBoard logs."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        ea = EventAccumulator(log_dir)
        ea.Reload()
        
        data = {}
        for tag in ea.Tags()['scalars']:
            scalar_events = ea.Scalars(tag)
            data[tag] = {
                'steps': [event.step for event in scalar_events],
                'values': [event.value for event in scalar_events]
            }
        return data
    except ImportError:
        print("TensorBoard not available for data loading")
        return {}


def load_experiment_results(experiment_dir):
    """Load results from all experiments in a directory."""
    results = {}
    
    for exp_path in glob.glob(os.path.join(experiment_dir, "*")):
        if not os.path.isdir(exp_path):
            continue
            
        exp_name = os.path.basename(exp_path)
        results[exp_name] = {}
        
        # Load results from each seed
        for seed_path in glob.glob(os.path.join(exp_path, "seed_*")):
            if not os.path.isdir(seed_path):
                continue
                
            seed = os.path.basename(seed_path)
            
            # Load configuration
            config_path = os.path.join(seed_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
            
            # Load TensorBoard data
            tb_data = load_tensorboard_data(seed_path)
            
            results[exp_name][seed] = {
                'config': config,
                'tensorboard_data': tb_data
            }
    
    return results


def compute_learning_curves(results, metric_name='train/Last100EpisodeScores'):
    """Compute learning curves for all experiments."""
    learning_curves = {}
    
    for exp_name, exp_data in results.items():
        curves = []
        
        for seed, seed_data in exp_data.items():
            tb_data = seed_data['tensorboard_data']
            if metric_name in tb_data:
                steps = tb_data[metric_name]['steps']
                values = tb_data[metric_name]['values']
                curves.append(pd.DataFrame({'step': steps, 'value': values, 'seed': seed}))
        
        if curves:
            learning_curves[exp_name] = pd.concat(curves, ignore_index=True)
    
    return learning_curves


def plot_learning_curves(learning_curves, metric_name='Score', save_path=None):
    """Plot learning curves with confidence intervals."""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(learning_curves)))
    
    for i, (exp_name, curve_data) in enumerate(learning_curves.items()):
        # Compute mean and std across seeds
        grouped = curve_data.groupby('step')['value'].agg(['mean', 'std', 'count']).reset_index()
        
        # Compute confidence intervals
        confidence = 0.95
        alpha = 1 - confidence
        grouped['ci'] = grouped['std'] / np.sqrt(grouped['count']) * stats.t.ppf(1 - alpha/2, grouped['count'] - 1)
        
        # Plot mean line
        plt.plot(grouped['step'], grouped['mean'], label=exp_name, color=colors[i], linewidth=2)
        
        # Plot confidence interval
        plt.fill_between(grouped['step'], 
                        grouped['mean'] - grouped['ci'], 
                        grouped['mean'] + grouped['ci'], 
                        alpha=0.2, color=colors[i])
    
    plt.xlabel('Training Step')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Learning Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def compute_sample_efficiency(learning_curves, threshold_scores=[10, 20, 50]):
    """Compute sample efficiency metrics."""
    efficiency_results = {}
    
    for exp_name, curve_data in learning_curves.items():
        efficiency_results[exp_name] = {}
        
        for threshold in threshold_scores:
            steps_to_threshold = []
            
            for seed in curve_data['seed'].unique():
                seed_data = curve_data[curve_data['seed'] == seed]
                
                # Find first step where score exceeds threshold
                above_threshold = seed_data[seed_data['value'] >= threshold]
                if not above_threshold.empty:
                    steps_to_threshold.append(above_threshold['step'].iloc[0])
                else:
                    steps_to_threshold.append(np.inf)  # Never reached threshold
            
            efficiency_results[exp_name][f'steps_to_{threshold}'] = {
                'mean': np.mean(steps_to_threshold),
                'std': np.std(steps_to_threshold),
                'success_rate': np.mean([s != np.inf for s in steps_to_threshold])
            }
    
    return efficiency_results


def compute_final_performance(learning_curves, final_steps=10000):
    """Compute final performance metrics."""
    final_performance = {}
    
    for exp_name, curve_data in learning_curves.items():
        final_scores = []
        
        for seed in curve_data['seed'].unique():
            seed_data = curve_data[curve_data['seed'] == seed]
            
            # Get scores from final steps
            final_data = seed_data[seed_data['step'] >= seed_data['step'].max() - final_steps]
            if not final_data.empty:
                final_scores.append(final_data['value'].mean())
        
        if final_scores:
            final_performance[exp_name] = {
                'mean': np.mean(final_scores),
                'std': np.std(final_scores),
                'scores': final_scores
            }
    
    return final_performance


def statistical_comparison(final_performance, baseline_name='baseline_no_shaping'):
    """Perform statistical comparisons against baseline."""
    if baseline_name not in final_performance:
        print(f"Baseline {baseline_name} not found in results")
        return {}
    
    baseline_scores = final_performance[baseline_name]['scores']
    comparisons = {}
    
    for exp_name, exp_data in final_performance.items():
        if exp_name == baseline_name:
            continue
            
        exp_scores = exp_data['scores']
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(exp_scores, baseline_scores)
        
        # Compute effect size (Cohen's d)
        pooled_std = np.sqrt(((len(exp_scores) - 1) * np.var(exp_scores, ddof=1) + 
                             (len(baseline_scores) - 1) * np.var(baseline_scores, ddof=1)) / 
                            (len(exp_scores) + len(baseline_scores) - 2))
        cohens_d = (np.mean(exp_scores) - np.mean(baseline_scores)) / pooled_std
        
        comparisons[exp_name] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05,
            'improvement': np.mean(exp_scores) > np.mean(baseline_scores)
        }
    
    return comparisons


def generate_report(results, learning_curves, efficiency_results, 
                   final_performance, statistical_comparisons, save_path=None):
    """Generate comprehensive analysis report."""
    report = "# Adaptive Reward Shaping Experimental Results\n\n"
    
    # Experiment overview
    report += "## Experiment Overview\n\n"
    report += f"Total experiments: {len(results)}\n"
    for exp_name in results.keys():
        num_seeds = len(results[exp_name])
        report += f"- {exp_name}: {num_seeds} seeds\n"
    report += "\n"
    
    # Final performance summary
    report += "## Final Performance Summary\n\n"
    report += "| Experiment | Mean Score | Std Dev | Improvement |\n"
    report += "|------------|------------|---------|-------------|\n"
    
    baseline_mean = final_performance.get('baseline_no_shaping', {}).get('mean', 0)
    
    for exp_name, perf_data in final_performance.items():
        mean_score = perf_data['mean']
        std_score = perf_data['std']
        improvement = ((mean_score - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0
        report += f"| {exp_name} | {mean_score:.2f} | {std_score:.2f} | {improvement:+.1f}% |\n"
    
    report += "\n"
    
    # Statistical significance
    report += "## Statistical Significance\n\n"
    report += "| Experiment | p-value | Cohen's d | Significant | Effect Size |\n"
    report += "|------------|---------|-----------|-------------|-------------|\n"
    
    for exp_name, comp_data in statistical_comparisons.items():
        p_val = comp_data['p_value']
        cohens_d = comp_data['cohens_d']
        significant = "Yes" if comp_data['significant'] else "No"
        
        if abs(cohens_d) < 0.2:
            effect_size = "Small"
        elif abs(cohens_d) < 0.5:
            effect_size = "Medium"
        else:
            effect_size = "Large"
            
        report += f"| {exp_name} | {p_val:.4f} | {cohens_d:.3f} | {significant} | {effect_size} |\n"
    
    report += "\n"
    
    # Sample efficiency
    report += "## Sample Efficiency\n\n"
    for threshold in [10, 20, 50]:
        report += f"### Steps to reach score {threshold}\n\n"
        report += "| Experiment | Mean Steps | Success Rate |\n"
        report += "|------------|------------|-------------|\n"
        
        for exp_name, eff_data in efficiency_results.items():
            threshold_key = f'steps_to_{threshold}'
            if threshold_key in eff_data:
                mean_steps = eff_data[threshold_key]['mean']
                success_rate = eff_data[threshold_key]['success_rate']
                
                if mean_steps == np.inf:
                    mean_steps_str = "Never"
                else:
                    mean_steps_str = f"{mean_steps:.0f}"
                    
                report += f"| {exp_name} | {mean_steps_str} | {success_rate:.1%} |\n"
        
        report += "\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze adaptive reward shaping results")
    parser.add_argument("experiment_dir", help="Directory containing experiment results")
    parser.add_argument("--output_dir", default="analysis", help="Output directory for analysis")
    parser.add_argument("--metric", default="train/Last100EpisodeScores", 
                       help="Metric to analyze")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print("Loading experimental results...")
    results = load_experiment_results(args.experiment_dir)
    
    if not results:
        print("No experimental results found!")
        return
    
    print(f"Loaded {len(results)} experiments")
    
    # Compute learning curves
    print("Computing learning curves...")
    learning_curves = compute_learning_curves(results, args.metric)
    
    # Plot learning curves
    print("Plotting learning curves...")
    plot_path = os.path.join(args.output_dir, "learning_curves.png")
    plot_learning_curves(learning_curves, save_path=plot_path)
    
    # Compute sample efficiency
    print("Computing sample efficiency...")
    efficiency_results = compute_sample_efficiency(learning_curves)
    
    # Compute final performance
    print("Computing final performance...")
    final_performance = compute_final_performance(learning_curves)
    
    # Statistical comparisons
    print("Performing statistical comparisons...")
    statistical_comparisons = statistical_comparison(final_performance)
    
    # Generate report
    print("Generating analysis report...")
    report_path = os.path.join(args.output_dir, "analysis_report.md")
    report = generate_report(results, learning_curves, efficiency_results,
                           final_performance, statistical_comparisons, report_path)
    
    print(f"Analysis complete! Results saved to {args.output_dir}")
    print("\nSummary:")
    print(report[:500] + "..." if len(report) > 500 else report)


if __name__ == "__main__":
    main()