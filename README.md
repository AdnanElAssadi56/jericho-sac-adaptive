# Adaptive Reward Shaping for Text-based Adventure Games

This repository implements **Adaptive Reward Shaping** techniques for Soft Actor-Critic (SAC) in text-based adventure games, extending the work from ["Learning to play text-based adventure games with maximum entropy reinforcement learning"](https://arxiv.org/abs/2302.10720).

## Research Overview

**Problem**: Text-based games suffer from sparse rewards, making RL training slow and unstable.

**Solution**: We propose adaptive reward shaping that dynamically adjusts the shaping coefficient α(t) during training, rather than using fixed reward shaping.

**Key Innovation**: Three adaptive schedulers that adjust reward shaping based on:
- **Time Decay**: Reduce shaping as training progresses
- **Sparsity-Triggered**: Boost shaping during reward-sparse periods  
- **Uncertainty-Informed**: Adjust based on policy entropy/uncertainty

## Quick Start

### 1. Install Dependencies
```bash
# Install PyTorch
conda install pytorch torchvision torchaudio -c pytorch

# Install other dependencies
pip install sentencepiece tensorboardX jericho

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

**Important**: The `unigram_8k.model` file is required for text tokenization. After extraction, you should see:
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
    --rom_path ./dependencies/z-machine-games-master/jericho-game-suite/zork1.z5 \
    --spm_path "./models/spm_models/unigram_8k.model" \
    --reward_shaping False \
    --max_steps 1000

# 2. SAC + Static Reward Shaping  
python src/train.py \
    --output_dir ./logging/validation/static_shaping \
    --rom_path ./dependencies/z-machine-games-master/jericho-game-suite/zork1.z5 \
    --spm_path "./models/spm_models/unigram_8k.model" \
    --reward_shaping True \
    --adaptive_shaping False \
    --max_steps 1000

# 3. SAC + Adaptive Reward Shaping (Our Method)
python src/train.py \
    --output_dir ./logging/validation/adaptive_shaping \
    --rom_path ./dependencies/z-machine-games-master/jericho-game-suite/zork1.z5 \
    --spm_path "./models/spm_models/unigram_8k.model" \
    --reward_shaping True \
    --adaptive_shaping True \
    --scheduler_type time_decay \
    --initial_alpha 1.0 \
    --decay_rate 0.001 \
    --max_steps 1000
```

**Full Multi-Seed Experiments:**
```bash
python run_experiments.py \
    --rom_path ./dependencies/z-machine-games-master/jericho-game-suite/zork1.z5 \
    --spm_path "./models/spm_models/unigram_8k.model" \
    --output_dir ./logging/experiments \
    --seeds 0 1 2 3 4 \
    --max_steps 50000
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

### Reward Shaping Parameters (your research)
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
│   └── models.py             # Neural network architectures
├── models/                   # SentencePiece models only
│   ├── spm_models.zip        # SentencePiece model archive
│   └── spm_models/           # Extracted SentencePiece models (after setup)
│       ├── unigram_8k.model  # SentencePiece model file
│       └── unigram_8k.vocab  # SentencePiece vocabulary file
├── run_experiments.py        # Multi-seed experiment runner
├── dependencies/             # Game files (setup required)
└── logging/                  # Experiment outputs (created during runs)
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

## Contact

For questions about the adaptive reward shaping implementation, please open an issue or contact the repository maintainer. 
