#!/usr/bin/env python3
"""
Quick script to check all final 50k step results
"""
import json
import os
from pathlib import Path

import sys
GAME = sys.argv[1] if len(sys.argv) > 1 else "zork1"
FINAL_DIR = f"logging/final/{GAME}/seed0"

methods = [
    'baseline_no_shaping',
    'baseline_static_shaping',
    'adaptive_time_decay_exp',
    'adaptive_time_decay_linear',
    'adaptive_time_decay_cosine',
    'adaptive_sparsity_triggered',
    'adaptive_sparsity_sensitive',
    'adaptive_uncertainty_informed'
]

print("="*80)
print(f"CHECKING 50K STEP RESULTS - {GAME.upper()} SEED0")
print("="*80)
print()

Path(FINAL_DIR).mkdir(parents=True, exist_ok=True)

results = []

for method in methods:
    method_dir = Path(FINAL_DIR) / method
    progress_file = method_dir / "progress.json"
    
    if not progress_file.exists():
        print(f"[X] {method}: progress.json NOT FOUND")
        continue
    
    # Count lines and get final score
    with open(progress_file, 'r') as f:
        lines = f.readlines()
        num_entries = len(lines)
        
        if num_entries == 0:
            print(f"❌ {method}: EMPTY FILE")
            continue
        
        # Get final score
        try:
            last_entry = json.loads(lines[-1].strip())
            final_score = last_entry['train/Last100EpisodeScores']
            
            # Get first score
            first_entry = json.loads(lines[0].strip())
            first_score = first_entry['train/Last100EpisodeScores']
            
            # Calculate steps (entries * 100)
            steps = num_entries * 100
            
            status = "[OK]" if num_entries >= 490 else "[!]"
            
            results.append({
                'method': method,
                'entries': num_entries,
                'steps': steps,
                'first_score': first_score,
                'final_score': final_score,
                'improvement': final_score - first_score,
                'status': status
            })
            
            print(f"{status} {method}")
            print(f"   Entries: {num_entries} | Steps: ~{steps}")
            print(f"   First: {first_score:.2f} | Final: {final_score:.2f} | Δ: {final_score - first_score:+.2f}")
            print()
            
        except Exception as e:
            print(f"[X] {method}: ERROR reading data - {e}")
            print()

print("="*80)
print("SUMMARY")
print("="*80)
print()

# Sort by final score
results.sort(key=lambda x: x['final_score'], reverse=True)

print(f"{'Rank':<6} {'Method':<40} {'Final Score':<12} {'Status'}")
print("-"*80)

for i, r in enumerate(results, 1):
    print(f"{i:<6} {r['method']:<40} {r['final_score']:<12.2f} {r['status']}")

print()
print("="*80)
print("KEY FINDINGS")
print("="*80)
print()

if results:
    best = results[0]
    baseline_no = next((r for r in results if r['method'] == 'baseline_no_shaping'), None)
    baseline_static = next((r for r in results if r['method'] == 'baseline_static_shaping'), None)
    
    print(f"Best Performer: {best['method']}")
    print(f"   Score: {best['final_score']:.2f}")
    
    if baseline_no:
        print(f"\nBaseline (No Shaping): {baseline_no['final_score']:.2f}")
        print(f"   Paper's result: 25.7")
        print(f"   Accuracy: {(baseline_no['final_score']/25.7)*100:.1f}%")
    
    if baseline_static:
        print(f"\nBaseline (Static Shaping): {baseline_static['final_score']:.2f}")
        print(f"   Paper's result: 36.0")
        print(f"   Accuracy: {(baseline_static['final_score']/36.0)*100:.1f}%")
    
    # Count adaptive methods beating static
    if baseline_static:
        adaptive_better = [r for r in results if 'adaptive' in r['method'] and r['final_score'] > baseline_static['final_score']]
        print(f"\nAdaptive methods beating static: {len(adaptive_better)}/{len([r for r in results if 'adaptive' in r['method']])}")
        for r in adaptive_better:
            improvement = ((r['final_score'] - baseline_static['final_score']) / baseline_static['final_score']) * 100
            print(f"   - {r['method']}: {r['final_score']:.2f} ({improvement:+.1f}%)")

print()
print("="*80)
print("READY FOR ITERATION 2?")
print("="*80)
print()

complete = len([r for r in results if r['entries'] >= 490])
total = len(methods)

if complete == total:
    print(f"ALL {total} METHODS COMPLETE!")
    print(f"Ready to run 2 more seeds + 2 more games")
    print(f"\nNext steps:")
    print(f"  1. Run seed1 and seed2 for Zork1")
    print(f"  2. Run all 3 seeds for Zork3")
    print(f"  3. Run all 3 seeds for Detective")
    print(f"  Total remaining: {total * 2 + total * 3 + total * 3} experiments")
else:
    print(f"{complete}/{total} methods complete")
    print(f"   Still running: {total - complete} experiments")

print()
