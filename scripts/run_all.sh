#!/bin/bash
# Run all analysis and figure generation scripts
# Usage: bash run_all.sh [game_name]

GAME=${1:-zork1}

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                          ║"
echo "║              Generating Paper Tables and Figures                         ║"
echo "║                                                                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Game: $GAME"
echo ""

# Step 1: Analyze results
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Analyzing results..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 analyze_results.py
if [ $? -ne 0 ]; then
    echo "❌ Error in analyze_results.py"
    exit 1
fi
echo ""

# Step 2: Generate main figures
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Generating main figures..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 generate_paper_figures.py
if [ $? -ne 0 ]; then
    echo "❌ Error in generate_paper_figures.py"
    exit 1
fi
echo ""

# Step 3: Generate α(t) evolution figure
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Generating α(t) evolution figure..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 generate_alpha_evolution.py
if [ $? -ne 0 ]; then
    echo "❌ Error in generate_alpha_evolution.py"
    exit 1
fi
echo ""

# Summary
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                          ║"
echo "║                          ✅ ALL COMPLETE ✅                             ║"
echo "║                                                                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Generated files:"
echo "  Data: results/data/${GAME}_results.pkl"
echo "  Figures:"
echo "     - results/figures/learning_curves.pdf"
echo "     - results/figures/scheduler_behavior.pdf"
echo "     - results/figures/final_comparison.pdf"
echo "     - results/figures/improvement_comparison.pdf"
echo "     - results/figures/alpha_evolution.pdf"
echo ""
