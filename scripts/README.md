# Analysis and Visualization Scripts

This directory contains all scripts for analyzing experimental results and generating visualizations.

## Scripts

### 1. `analyze_results.py`
**Purpose**: Analyze experimental results and generate LaTeX tables

**Usage**:
```bash
python analyze_results.py
```

**Outputs**:
- Prints statistics for all methods
- Generates LaTeX-formatted tables
- Saves processed data to `../results/data/zork1_results.pkl`

**What it does**:
- Loads progress.json files from `logging/final/zork1/seed{0,1,3}/`
- Calculates mean ± std across 3 seeds
- Computes improvement percentages vs static baseline
- Generates per-seed breakdown
- Compares to Li et al. (2023) baseline

**Modify for new games**:
```python
# Line 11: Change game name
GAME = "zork3"  # or "detective", "pentari", etc.

# Line 12: Add/remove seeds
SEEDS = ["seed0", "seed1", "seed2", "seed3"]  # Add more seeds
```

### 2. `generate_paper_figures.py`
**Purpose**: Generate all main figures for the paper

**Usage**:
```bash
python generate_paper_figures.py
```

**Outputs**:
- `../results/figures/learning_curves.pdf` - Learning curves with mean ± std
- `../results/figures/scheduler_behavior.pdf` - Time-based scheduler visualization
- `../results/figures/final_comparison.pdf` - Bar chart of final performance
- `../results/figures/improvement_comparison.pdf` - Improvement percentages

**What it does**:
- Loads all progress.json files
- Computes mean and std across seeds
- Creates publication-quality figures (300 DPI)
- Generates both PDF and PNG versions

**Modify for new games**:
```python
# Line 23: Change game name
GAME = "zork3"

# Line 24: Add/remove seeds
SEEDS = ["seed0", "seed1", "seed2", "seed3"]
```

### 3. `generate_alpha_evolution.py`
**Purpose**: Generate α(t) evolution visualization (Figure 2 in paper)

**Usage**:
```bash
python generate_alpha_evolution.py
```

**Outputs**:
- `results/figures/alpha_evolution.pdf` - 4-panel visualization of α(t)
  - (a) Time-based schedulers
  - (b) Sparsity-triggered behavior
  - (c) Entropy-informed with policy entropy
  - (d) Comparison of best methods

**What it does**:
- Simulates scheduler behavior over 50k steps
- Shows reactive adaptation for sparsity-based
- Visualizes entropy tracking for uncertainty-based
- Compares all adaptive methods

**Note**: This uses simulated data for visualization. For actual α(t) from logs, modify to load from log files.

## Quick Start

To regenerate all analysis and figures:

```bash
# From project root, navigate to scripts
cd scripts

# 1. Analyze results and generate tables
python analyze_results.py > results_summary.txt

# 2. Generate all figures
python generate_paper_figures.py

# 3. Generate α(t) evolution figure
python generate_alpha_evolution.py

# 4. Multi-game analysis
python analyze_all_games.py

# 5. Check outputs
ls ../results/figures/  # Should show PDF and PNG files
ls ../results/data/     # Should show .pkl files
```

## Adding New Games

To add results from a new game (e.g., Zork3):

1. **Run experiments** and save to `logging/final/zork3/seed{0,1,2}/`

2. **Modify scripts**:
```python
# In analyze_results.py and generate_paper_figures.py
GAME = "zork3"  # Line 11 or 23
```

3. **Run analysis**:
```bash
cd scripts
python analyze_results.py
python generate_paper_figures.py
```

## Adding More Seeds

To add more seeds (e.g., seed4, seed5):

1. **Run experiments** with new seeds

2. **Modify scripts**:
```python
# In analyze_results.py and generate_paper_figures.py
SEEDS = ["seed0", "seed1", "seed2", "seed3", "seed4"]  # Add new seeds
```

3. **Run analysis**:
```bash
cd scripts
python analyze_results.py
python generate_paper_figures.py
```

## Available Scripts

| Script | Purpose | Outputs |
|--------|---------|---------|
| `analyze_results.py` | Single game statistical analysis | LaTeX tables, data files |
| `analyze_all_games.py` | Multi-game comparison | Cross-game statistics |
| `analyze_multi_game.py` | Multi-game analysis (3 games) | LaTeX tables |
| `analyze_convergence.py` | Convergence analysis | Sample efficiency metrics |
| `generate_paper_figures.py` | Main visualizations | Learning curves, comparisons |
| `generate_alpha_evolution.py` | α(t) visualization | Scheduler behavior plots |
| `generate_main_figures.py` | Multi-game figures | Training progress, heatmaps |
| `generate_tradeoff_figure.py` | Efficiency tradeoff | Sample efficiency vs performance |
| `create_figures.py` | Publication figures | Comprehensive visualizations |
| `check_final_results.py` | Verify experiment completion | Status report |

## Dependencies

```bash
pip install numpy matplotlib scipy
```

## Output Structure

```
results/
├── figures/                           # All generated visualizations
│   ├── learning_curves.pdf/png
│   ├── scheduler_behavior.pdf/png
│   ├── final_comparison.pdf/png
│   ├── improvement_comparison.pdf/png
│   ├── alpha_evolution.pdf/png
│   ├── training_progress.pdf/png
│   └── performance_heatmap.pdf/png
└── data/                              # Processed data
    ├── zork1_results.pkl
    ├── multi_game_results.pkl
    └── all_games_results.pkl
```

## Customization

All scripts use consistent method names and colors. To add new methods:

1. Add to `METHODS` list in each script
2. Add to `METHOD_NAMES` dictionary
3. Add to `COLORS` dictionary (for visualization scripts)
4. Re-run analysis

## Troubleshooting

### "File not found" errors
- Check that `logging/final/zork1/seed{0,1,3}/` exists
- Verify progress.json files are present
- Check GAME and SEEDS variables match the directory structure

### "No module named 'scipy'"
```bash
pip install scipy
```
