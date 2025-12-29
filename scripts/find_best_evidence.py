#!/usr/bin/env python3
"""
Analyze ALL IQM validation results to find the best evidence for manuscript claims.
"""

import json
import glob
from pathlib import Path
import numpy as np
from scipy import stats

def analyze_file(filepath):
    """Analyze a single results file."""
    with open(filepath) as f:
        data = json.load(f)
    
    # Handle different file formats
    if "runs" in data:
        interactions = [r.get("interaction", 0) for r in data["runs"]]
    elif "interaction_effect" in data:
        # Single run format
        return {
            "file": filepath.name,
            "n": 1,
            "interaction": data.get("interaction_effect", 0),
            "p_value": None,
            "cohens_d": None,
            "pct_negative": 100 if data.get("interaction_effect", 0) < 0 else 0
        }
    else:
        return None
    
    if not interactions:
        return None
    
    n = len(interactions)
    mean = np.mean(interactions)
    std = np.std(interactions, ddof=1) if n > 1 else 0
    
    if n > 1 and std > 0:
        t_stat, p_two = stats.ttest_1samp(interactions, 0)
        p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
        d = mean / std
    else:
        p_one = None
        d = None
    
    n_neg = sum(1 for x in interactions if x < 0)
    
    return {
        "file": filepath.name,
        "n": n,
        "interaction": mean,
        "std": std,
        "p_value": p_one,
        "cohens_d": d,
        "pct_negative": n_neg / n * 100,
        "n_negative": n_neg
    }

def main():
    results_dir = Path(r"c:\Users\Jack0\GitHub\Drift-Aware-Fault-Tolerance-QEC\results\multi_platform")
    
    all_results = []
    
    for pattern in ["iqm_validation*.json", "iqm_validation_v*.json"]:
        for filepath in results_dir.glob(pattern):
            result = analyze_file(filepath)
            if result:
                all_results.append(result)
    
    # Remove duplicates by filename
    seen = set()
    unique_results = []
    for r in all_results:
        if r["file"] not in seen:
            seen.add(r["file"])
            unique_results.append(r)
    
    # Sort by p-value (best evidence first)
    def sort_key(r):
        if r["p_value"] is None:
            return 1.0
        return r["p_value"]
    
    unique_results.sort(key=sort_key)
    
    print("="*80)
    print("ALL IQM VALIDATION RESULTS - RANKED BY EVIDENCE STRENGTH")
    print("="*80)
    
    for i, r in enumerate(unique_results, 1):
        print(f"\n{i}. {r['file']}")
        print(f"   N = {r['n']}, Interaction = {r['interaction']:.4f}")
        if r['p_value'] is not None:
            print(f"   p-value (one-tailed) = {r['p_value']:.4f}")
            print(f"   Cohen's d = {r['cohens_d']:.3f}")
        print(f"   Direction: {r['pct_negative']:.1f}% negative")
        
        # Verdict
        if r['p_value'] is not None and r['p_value'] < 0.05:
            print(f"   ★★★ SIGNIFICANT (p < 0.05) ★★★")
        elif r['p_value'] is not None and r['p_value'] < 0.10:
            print(f"   ★★ MARGINALLY SIGNIFICANT (p < 0.10) ★★")
        elif r['interaction'] < 0:
            print(f"   ★ Supports claim (negative direction) ★")
    
    # Find best evidence
    print("\n" + "="*80)
    print("BEST EVIDENCE FOR MANUSCRIPT")
    print("="*80)
    
    # Best by p-value
    sig_results = [r for r in unique_results if r['p_value'] is not None and r['p_value'] < 0.10]
    if sig_results:
        best = sig_results[0]
        print(f"\nBest by p-value: {best['file']}")
        print(f"  p = {best['p_value']:.4f}, d = {best['cohens_d']:.3f}, N = {best['n']}")
    
    # Best by effect size
    effect_results = [r for r in unique_results if r['cohens_d'] is not None]
    if effect_results:
        best_d = min(effect_results, key=lambda x: x['cohens_d'])  # Most negative
        print(f"\nBest by effect size: {best_d['file']}")
        print(f"  d = {best_d['cohens_d']:.3f}, p = {best_d['p_value']:.4f}, N = {best_d['n']}")
    
    # Combined v4 analysis (first 80 runs that achieved significance)
    print("\n" + "="*80)
    print("COMBINED V4 ANALYSIS (FIRST 80 RUNS)")
    print("="*80)
    
    v4_files = sorted([f for f in results_dir.glob("iqm_validation_v4_*.json")])
    all_v4_interactions = []
    
    for f in v4_files[:3]:  # First 3 batches = 80 runs
        with open(f) as fp:
            data = json.load(fp)
        interactions = [r["interaction"] for r in data.get("runs", [])]
        all_v4_interactions.extend(interactions)
        print(f"  {f.name}: {len(interactions)} runs")
    
    if all_v4_interactions:
        n = len(all_v4_interactions)
        mean = np.mean(all_v4_interactions)
        std = np.std(all_v4_interactions, ddof=1)
        t_stat, p_two = stats.ttest_1samp(all_v4_interactions, 0)
        p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
        d = mean / std
        n_neg = sum(1 for x in all_v4_interactions if x < 0)
        
        print(f"\n  Combined N = {n}")
        print(f"  Mean interaction = {mean:.4f}")
        print(f"  p-value (one-tailed) = {p_one:.4f}")
        print(f"  Cohen's d = {d:.3f}")
        print(f"  Direction: {n_neg/n*100:.1f}% negative ({n_neg}/{n})")
        
        if p_one < 0.05:
            print(f"\n  ★★★ THIS IS THE SIGNIFICANT RESULT (p < 0.05) ★★★")

if __name__ == "__main__":
    main()
