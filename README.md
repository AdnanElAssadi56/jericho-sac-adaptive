# Adaptive Reward Shaping for Text-based Adventure Games

This repository implements **Adaptive Reward Shaping** techniques for Soft Actor-Critic (SAC) in text-based adventure games, extending the work from ["Learning to play text-based adventure games with maximum entropy reinforcement learning"](https://arxiv.org/abs/2302.10720).

## Research Overview

**Problem**: Text-based games suffer from sparse rewards, making RL training slow and unstable.

**Solution**: We propose adaptive reward shaping that dynamically adjusts the shaping coefficient α(t) during training, rather than using fixed reward shaping.

**Key Innovation**: Six adaptive schedulers that adjust reward shaping based on:

### **Time-Based Adaptation (Curriculum Learning)**
- **Exponential Decay**: α(t) = α₀ × exp(-λt) - Smooth exponential reduction
- **Linear Decay**: α(t) = α₀ × (1 - t/T) - Linear reduction over time  
- **Cosine Decay**: α(t) = α₀ × (1 + cos(πt/T))/2 - Smooth start and end

### **Game-Aware Adaptation (Difficulty Response)**
- **Sparsity-Triggered**: Boost α when no rewards for N steps - Responds to game difficulty
- **Sparsity-Sensitive**: Lower threshold (25 vs 50 steps) - More responsive to sparse periods

### **Agent-Aware Adaptation (Self-Monitoring)**
- **Uncertainty-Informed**: α based on policy entropy - Uses agent's confidence level

**Research Hypothesis**: Adaptive > Static > Baseline, with different schedulers excelling in different scenarios

## Quick Start

### 1. Install Dependencies
```bash
# Install PyTorch
conda install pytorch torchvision torchaudio -c pytorch

# Install other dependencies
pip install matplotlib scipy sentencepiece wandb tensorboardX jericho

# Install spaCy model for Jericho
python -m spacy download en_core_web_sm

```

### 2. Setup Game Files
```bash
# Create dependencies directory
mkdir dependencies
cd dependencies

# Download Jericho game suite
wget https://github.com/BYU-PCCL/z-machine-games/archive/master.zip
unzip master.zip
rm master.zip
cd ..
```

### 3. Extract SentencePiece Model
```bash
# Navigate to the models directory
cd models

# Extract the SentencePiece model (creates spm_models/ subdirectory)
unzip spm_models.zip

# Verify the model files exist
ls -la spm_models/

# Return to project root
cd ..
```

**Important**: The `unigram_8k.model` file is required for text tokenization. After extraction, the following files will be present:
- `models/spm_models/unigram_8k.model` - The SentencePiece model file
- `models/spm_models/unigram_8k.vocab` - The vocabulary file

### 3. Run Experiments

**Basic Run Command:**
```bash
# Basic usage (from original implementation)
python src/train.py \
    --output_dir 'path' \
    --rom_path 'path/game' \
    --spm_path 'path'
```

**Quick Validation (1,000 steps each):**
```bash
# 1. SAC Baseline (no reward shaping)
python src/train.py \    
    --output_dir ./logging/validation/baseline_sac \    
    --rom_path ./dependencies/z-machine-games-master/jericho-game-suite/zork1.z5 \    --spm_path "./models/spm_models/unigram_8k.model" \    
    --reward_shaping False \    
    --max_steps 10000

# 2. SAC + Static Reward Shaping  
python src/train.py \
    --output_dir ./logging/validation/static_shaping \
    --rom_path ./dependencies/z-machine-games-master/jericho-game-suite/zork1.z5 \
    --spm_path "./models/spm_models/unigram_8k.model" \
    --reward_shaping True \
    --adaptive_shaping False \
    --max_steps 1000

# 3. SAC + Adaptive Time Decay (Our Method)
python src/train.py \
    --output_dir ./logging/validation/adaptive_time_decay \
    --rom_path ./dependencies/z-machine-games-master/jericho-game-suite/zork1.z5 \
    --spm_path "./models/spm_models/unigram_8k.model" \
    --reward_shaping True \
    --adaptive_shaping True \
    --scheduler_type time_decay \
    --initial_alpha 1.0 \
    --decay_rate 0.001 \
    --max_steps 1000

# 4. SAC + Adaptive Sparsity-Triggered (Our Method)
python src/train.py \
    --output_dir ./logging/validation/adaptive_sparsity \
    --rom_path ./dependencies/z-machine-games-master/jericho-game-suite/zork1.z5 \
    --spm_path "./models/spm_models/unigram_8k.model" \
    --reward_shaping True \
    --adaptive_shaping True \
    --scheduler_type sparsity_triggered \
    --initial_alpha 1.0 \
    --sparsity_threshold 50 \
    --max_steps 1000

# 5. SAC + Adaptive Uncertainty-Informed (Our Method)
python src/train.py \
    --output_dir ./logging/validation/adaptive_uncertainty \
    --rom_path ./dependencies/z-machine-games-master/jericho-game-suite/zork1.z5 \
    --spm_path "./models/spm_models/unigram_8k.model" \
    --reward_shaping True \
    --adaptive_shaping True \
    --scheduler_type uncertainty_informed \
    --initial_alpha 1.0 \
    --entropy_threshold 0.5 \
    --max_steps 1000
```

**Full Multi-Seed Experiments:**

Use the tmux scripts in `run_experiments/` to run all methods across multiple seeds:

```bash
# Run all experiments for Zork1 (3 seeds)
bash run_experiments/run_zork1_tmux.sh

# Run experiments for other games
bash run_experiments/run_detective_tmux.sh
bash run_experiments/run_pentari_tmux.sh
bash run_experiments/run_adventureland_tmux.sh
```

## Adaptive Schedulers

### Time Decay Scheduler
```bash
--scheduler_type time_decay --initial_alpha 1.0 --decay_rate 0.001
```
- **Exponential**: α(t) = α₀ × exp(-λt)
- **Linear**: α(t) = α₀ × (1 - t/T)  
- **Cosine**: α(t) = α₀ × (1 + cos(πt/T))/2

### Sparsity-Triggered Scheduler
```bash
--scheduler_type sparsity_triggered --sparsity_threshold 50 --boost_factor 2.0
```
- Monitors reward-free episodes
- Boosts shaping when sparsity threshold exceeded

### Uncertainty-Informed Scheduler  
```bash
--scheduler_type uncertainty_informed --entropy_threshold 0.5
```
- Based on policy entropy and Q-value variance
- High uncertainty → more shaping

## Key Parameters

### Core SAC Parameters (from original paper)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 32 | Batch size for training |
| `--learning_rate` | 0.0003 | Learning rate for actor and critic |
| `--gamma` | 0.9 | Discount factor |
| `--num_envs` | 8 | Parallel environments |
| `--env_step_limit` | 100 | Max steps per episode |
| `--embedding_dim` | 128 | Text embedding dimension |
| `--hidden_dim` | 128 | Hidden layer dimension |

### Adaptive Reward Shaping Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--reward_shaping` | False | Enable reward shaping |
| `--adaptive_shaping` | False | Enable adaptive shaping |
| `--scheduler_type` | time_decay | Scheduler type |
| `--initial_alpha` | 1.0 | Initial shaping coefficient |
| `--decay_rate` | 0.001 | Decay rate for time-based schedulers |
| `--sparsity_threshold` | 50 | Steps without reward to trigger boost |

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_steps` | 50000 | Maximum training steps |
| `--checkpoint_freq` | 5000 | Save checkpoint frequency |
| `--log_freq` | 100 | Logging frequency |

## Research Phases

### Phase 1: Quick Validation (1,000 steps each)
**Goal**: Verify all three approaches work and show learning
**Time**: 1-2 hours total
**Success**: Agent scores should improve over episodes

### Phase 2: Baseline Reproduction (50,000 steps)
**Goal**: Reproduce paper results for SAC baseline
**Time**: 6-12 hours per experiment
**Success**: Results should match paper's SAC performance

### Phase 3: Full Multi-Seed Experiments  
**Goal**: Statistical comparison of all three approaches
**Time**: 45-75 hours total (5 seeds × 3 approaches)
**Success**: Adaptive shaping outperforms static shaping and baseline

## Expected Results

**Research Hypothesis**: SAC + Adaptive Shaping > SAC + Static Shaping > SAC Baseline

**Why Adaptive Should Win**:
1. **Early Training**: More guidance when agent is learning
2. **Late Training**: Less interference as agent improves  
3. **Game Adaptation**: Automatically adjusts to different game characteristics

## Troubleshooting

### Common Issues
- **Missing SentencePiece model**: Extract `spm_models.zip` first
- **Game file not found**: Check path to `.z5` files in dependencies
- **CUDA out of memory**: Reduce `--batch_size` to 16 or 8
- **Slow training**: Ensure GPU is detected, reduce `--num_envs`

### Performance Expectations
- **GPU**: ~2-4 hours per 50k step experiment
- **CPU**: ~6-12 hours per 50k step experiment
- **Learning**: Agent scores should improve within first 1000 steps

### Output Organization
All experiment outputs are organized in the `logging/` directory to keep the main project clean:
- `logging/validation/` - Quick validation runs (1,000 steps)
- `logging/experiments/` - Full multi-seed experiments (50,000 steps)
- `logging/baseline_reproduction/` - Baseline reproduction runs
- Each run creates subdirectories with logs, checkpoints, and TensorBoard files

## Repository Structure

```
├── src/                      # Source code
│   ├── train.py              # Main training script
│   ├── sac_rs.py             # SAC algorithm with reward shaping
│   ├── adaptive_shaping.py   # Adaptive shaping schedulers
│   ├── models.py             # Neural network architectures
│   ├── env.py                # Environment wrapper
│   ├── logger.py             # Logging utilities
│   ├── memory.py             # Replay buffer
│   └── utils.py              # Utility functions
├── scripts/                  # Analysis and visualization scripts
│   ├── analyze_results.py    # Statistical analysis
│   ├── analyze_all_games.py  # Multi-game analysis
│   ├── generate_paper_figures.py  # Generate figures
│   ├── generate_alpha_evolution.py  # α(t) visualization
│   └── README.md             # Scripts documentation
├── run_experiments/          # Experiment runners
│   ├── run_zork1_tmux.sh     # Run Zork1 experiments
│   ├── run_detective_tmux.sh # Run Detective experiments
│   ├── run_pentari_tmux.sh   # Run Pentari experiments
│   └── run_adventureland_tmux.sh  # Run Adventureland experiments
├── models/                   # SentencePiece models
│   └── spm_models/           # Extracted models (after setup)
│       ├── unigram_8k.model  # SentencePiece model file
│       └── unigram_8k.vocab  # SentencePiece vocabulary file
├── results/                  # Processed results and figures
│   ├── figures/              # Generated visualizations
│   ├── data/                 # Processed data (.pkl files)
│   └── README.md             # Results documentation
├── dependencies/             # Game files (gitignored, setup required)
│   └── z-machine-games-master/  # Jericho game suite
└── logging/                  # Raw experiment outputs (gitignored)
```

## References

This implementation builds upon several key works:

**Original SAC Paper:**
```bibtex
@article{li2023learning,
  title={Learning to play text-based adventure games with maximum entropy reinforcement learning},
  author={Li, Weichen and Devidze, Rati and Fellenz, Sophie},
  journal={arXiv preprint arXiv:2302.10720},
  year={2023}
}
```

**Jericho Environment:**
```bibtex
@article{hausknecht2020interactive,
  title={Interactive fiction games: A colossal adventure},
  author={Hausknecht, Matthew and Ammanabrolu, Prithviraj and C{\^o}t{\'e}, Marc-Alexandre and Yuan, Xinyi},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  pages={7903--7910},
  year={2020}
}
```

**DRRN and CALM (baseline implementations):**
```bibtex
@article{yao2020keep,
  title={Keep CALM and explore: Language models for action generation in text-based games},
  author={Yao, Shunyu and Rao, Rohan and Hausknecht, Matthew and Narasimhan, Karthik},
  journal={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
  pages={8736--8754},
  year={2020}
}
```

**Environment Source:** Our experiments use the publicly accessible [Jericho environment](https://github.com/microsoft/jericho) that provides the interface for playing all games in our experiments.

## Quick Validation Commands (10k Steps)

For rapid testing and validation of different adaptive reward shaping methods, use these commands with 10,000 training steps. All commands use the following default parameters unless specified:

### Default Parameters Used:
- `--num_envs 8` (8 parallel environments)
- `--batch_size 32`
- `--gamma 0.9` (discount factor)
- `--embedding_dim 128`
- `--hidden_dim 128`
- `--log_freq 100` (log every 100 steps)
- `--checkpoint_freq 5000`
- `--env_step_limit 100` (max steps per episode)
- `--memory_size 100000` (replay buffer size)
- `--seed 0`
- `--tensorboard 1 --wandb 0` (enable TensorBoard, disable W&B)

### 1. Baseline (No Shaping)
```bash
python src/train.py --output_dir logging/validation_10k/baseline_no_shaping --rom_path ./dependencies/z-machine-games-master/jericho-game-suite/zork1.z5 --spm_path ./models/spm_models/unigram_8k.model --max_steps 10000 --reward_shaping False --adaptive_shaping False --seed 0 --tensorboard 1 --wandb 0
```

### 2. Baseline (Static Shaping)
```bash
python src/train.py --output_dir logging/validation_10k/baseline_static_shaping --rom_path ./dependencies/z-machine-games-master/jericho-game-suite/zork1.z5 --spm_path ./models/spm_models/unigram_8k.model --max_steps 10000 --reward_shaping True --adaptive_shaping False --seed 0 --tensorboard 1 --wandb 0
```

### 3. Adaptive Time Decay (Exponential)
```bash
python src/train.py --output_dir logging/validation_10k/adaptive_time_decay_exp --rom_path ./dependencies/z-machine-games-master/jericho-game-suite/zork1.z5 --spm_path ./models/spm_models/unigram_8k.model --max_steps 10000 --reward_shaping True --adaptive_shaping True --scheduler_type time_decay --initial_alpha 1.0 --decay_rate 0.0002 --decay_type exponential --seed 0 --tensorboard 1 --wandb 0
```

### 4. Adaptive Time Decay (Linear)
```bash
python src/train.py --output_dir logging/validation_10k/adaptive_time_decay_linear --rom_path ./dependencies/z-machine-games-master/jericho-game-suite/zork1.z5 --spm_path ./models/spm_models/unigram_8k.model --max_steps 10000 --reward_shaping True --adaptive_shaping True --scheduler_type time_decay --initial_alpha 1.0 --decay_rate 0.001 --decay_type linear --seed 0 --tensorboard 1 --wandb 0
```

### 5. Adaptive Time Decay (Cosine)
```bash
python src/train.py --output_dir logging/validation_10k/adaptive_time_decay_cosine --rom_path ./dependencies/z-machine-games-master/jericho-game-suite/zork1.z5 --spm_path ./models/spm_models/unigram_8k.model --max_steps 10000 --reward_shaping True --adaptive_shaping True --scheduler_type time_decay --initial_alpha 1.0 --decay_rate 0.001 --decay_type cosine --seed 0 --tensorboard 1 --wandb 0
```

### 6. Adaptive Sparsity Triggered (threshold=50)
```bash
python src/train.py --output_dir logging/validation_10k/adaptive_sparsity_triggered --rom_path ./dependencies/z-machine-games-master/jericho-game-suite/zork1.z5 --spm_path ./models/spm_models/unigram_8k.model --max_steps 10000 --reward_shaping True --adaptive_shaping True --scheduler_type sparsity_triggered --initial_alpha 1.0 --sparsity_threshold 50 --seed 0 --tensorboard 1 --wandb 0
```

### 7. Adaptive Sparsity Sensitive (threshold=25)
```bash
python src/train.py --output_dir logging/validation_10k/adaptive_sparsity_sensitive --rom_path ./dependencies/z-machine-games-master/jericho-game-suite/zork1.z5 --spm_path ./models/spm_models/unigram_8k.model --max_steps 10000 --reward_shaping True --adaptive_shaping True --scheduler_type sparsity_triggered --initial_alpha 1.0 --sparsity_threshold 25 --seed 0 --tensorboard 1 --wandb 0
```

### 8. Adaptive Uncertainty Informed
```bash
python src/train.py --output_dir logging/validation_10k/adaptive_uncertainty_informed --rom_path ./dependencies/z-machine-games-master/jericho-game-suite/zork1.z5 --spm_path ./models/spm_models/unigram_8k.model --max_steps 10000 --reward_shaping True --adaptive_shaping True --scheduler_type uncertainty_informed --initial_alpha 1.0 --entropy_threshold 0.5 --uncertainty_window 50 --min_alpha 0.1 --max_alpha 2.0 --seed 0 --tensorboard 1 --wandb 0
```

### Monitoring Progress
```bash
# Check progress of any experiment
tail -f logging/validation_10k/[experiment_name]/progress.csv

# Use TensorBoard for real-time monitoring
tensorboard --logdir logging/validation_10k/

# Analyze results after completion
cd scripts
python analyze_results.py
python generate_paper_figures.py
```

### Expected Runtime
Each 10k step experiment takes approximately **2-4 hours** depending on hardware. Multiple experiments can be run in parallel if sufficient resources are available.

### Interpreting Results
- **Look for trends** rather than absolute scores at 10k steps
- **Compare relative performance** between methods
- **Methods showing improvement** at 10k are likely to perform better at full 50k training
- **Focus on learning curves** and stability, not just final scores

## Analyzing Results

After running experiments, use the analysis scripts to generate statistics and visualizations:

```bash
cd scripts

# Analyze single game results
python analyze_results.py

# Analyze multiple games
python analyze_all_games.py

# Generate figures
python generate_paper_figures.py
python generate_alpha_evolution.py

# Or run all analysis at once
bash run_all.sh
```

All outputs are saved to `results/figures/` and `results/data/`.

## Contact

For questions about the adaptive reward shaping implementation, please open an issue or contact the repository maintainer. 
