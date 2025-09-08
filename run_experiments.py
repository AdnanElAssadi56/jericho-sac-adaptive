#!/usr/bin/env python3
"""
Experimental runner for adaptive reward shaping research.
Author: Adnan El Assadi
"""

import os
import subprocess
import argparse
import json
from itertools import product
import time


def create_experiment_config(base_config, experiment_params):
    """Create experiment configuration by merging base config with experiment params."""
    config = base_config.copy()
    config.update(experiment_params)
    return config


def run_single_experiment(config, experiment_name, seed, output_base_dir):
    """Run a single experimental configuration."""
    # Create output directory
    output_dir = os.path.join(output_base_dir, experiment_name, f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        "python", "src/train.py",
        "--output_dir", output_dir,
        "--seed", str(seed),
        "--tensorboard", "1",
        "--wandb", "1" if config.get('use_wandb', False) else "0"
    ]
    
    # Add configuration parameters
    for key, value in config.items():
        if key not in ['use_wandb', 'description']:
            cmd.extend([f"--{key}", str(value)])
    
    print(f"Running experiment: {experiment_name}, seed: {seed}")
    print(f"Command: {' '.join(cmd)}")
    
    # Save configuration
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run experiment
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600*6)  # 6 hour timeout
        
        # Save logs
        with open(os.path.join(output_dir, "stdout.log"), 'w') as f:
            f.write(result.stdout)
        with open(os.path.join(output_dir, "stderr.log"), 'w') as f:
            f.write(result.stderr)
            
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"Experiment {experiment_name} (seed {seed}) timed out")
        return False
    except Exception as e:
        print(f"Error running experiment {experiment_name} (seed {seed}): {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run adaptive reward shaping experiments")
    parser.add_argument("--rom_path", required=True, help="Path to game ROM")
    parser.add_argument("--spm_path", required=True, help="Path to SentencePiece model")
    parser.add_argument("--output_dir", default="logging/experiments", help="Base output directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], 
                       help="Random seeds to use")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum training steps")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Base configuration
    base_config = {
        "rom_path": args.rom_path,
        "spm_path": args.spm_path,
        "max_steps": args.max_steps,
        "env_step_limit": 100,
        "num_envs": 8,
        "batch_size": 32,
        "memory_size": 100000,
        "learning_rate": 0.0001,
        "gamma": 0.9,
        "embedding_dim": 128,
        "hidden_dim": 128,
        "log_freq": 100,
        "checkpoint_freq": 5000,
        "enable_diagnostics": True,
        "use_wandb": args.use_wandb
    }
    
    # Define experimental configurations
    experiments = {
        # Baseline experiments
        "baseline_no_shaping": {
            "reward_shaping": False,
            "adaptive_shaping": False,
            "description": "Baseline SAC without reward shaping"
        },
        
        "baseline_static_shaping": {
            "reward_shaping": True,
            "adaptive_shaping": False,
            "description": "SAC with static reward shaping"
        },
        
        # Time-decay experiments
        "adaptive_time_decay_exp": {
            "reward_shaping": True,
            "adaptive_shaping": True,
            "scheduler_type": "time_decay",
            "initial_alpha": 1.0,
            "decay_rate": 0.001,
            "description": "Adaptive shaping with exponential time decay"
        },
        
        "adaptive_time_decay_linear": {
            "reward_shaping": True,
            "adaptive_shaping": True,
            "scheduler_type": "time_decay",
            "initial_alpha": 1.0,
            "decay_rate": 0.001,
            "description": "Adaptive shaping with linear time decay"
        },
        
        # Sparsity-triggered experiments
        "adaptive_sparsity_triggered": {
            "reward_shaping": True,
            "adaptive_shaping": True,
            "scheduler_type": "sparsity_triggered",
            "initial_alpha": 1.0,
            "sparsity_threshold": 50,
            "description": "Adaptive shaping triggered by reward sparsity"
        },
        
        "adaptive_sparsity_sensitive": {
            "reward_shaping": True,
            "adaptive_shaping": True,
            "scheduler_type": "sparsity_triggered",
            "initial_alpha": 1.0,
            "sparsity_threshold": 25,
            "description": "Adaptive shaping with sensitive sparsity threshold"
        },
        
        # Uncertainty-informed experiments
        "adaptive_uncertainty_informed": {
            "reward_shaping": True,
            "adaptive_shaping": True,
            "scheduler_type": "uncertainty_informed",
            "initial_alpha": 1.0,
            "description": "Adaptive shaping based on policy uncertainty"
        }
    }
    
    # Run experiments
    results = {}
    total_experiments = len(experiments) * len(args.seeds)
    current_experiment = 0
    
    for exp_name, exp_config in experiments.items():
        results[exp_name] = []
        
        for seed in args.seeds:
            current_experiment += 1
            print(f"\n=== Experiment {current_experiment}/{total_experiments} ===")
            
            # Create full configuration
            full_config = create_experiment_config(base_config, exp_config)
            
            # Run experiment
            success = run_single_experiment(full_config, exp_name, seed, args.output_dir)
            results[exp_name].append(success)
            
            print(f"Experiment {exp_name} (seed {seed}): {'SUCCESS' if success else 'FAILED'}")
            
            # Brief pause between experiments
            time.sleep(2)
    
    # Print summary
    print("\n=== EXPERIMENT SUMMARY ===")
    for exp_name, exp_results in results.items():
        success_rate = sum(exp_results) / len(exp_results)
        print(f"{exp_name}: {success_rate:.1%} success rate ({sum(exp_results)}/{len(exp_results)})")
    
    # Save results summary
    summary_path = os.path.join(args.output_dir, "experiment_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nExperiment summary saved to: {summary_path}")


if __name__ == "__main__":
    main()