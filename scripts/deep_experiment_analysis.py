#!/usr/bin/env python3
"""
DEEP COMPREHENSIVE ANALYSIS OF ALL EXPERIMENTAL DATA

This script performs thorough analysis of:
1. ALL files in multi_platform/ (47 files)
2. ALL files in hardware_validation/
3. ALL files in ibm_experiments/

Extracts:
- Every quantum task ID (Braket ARNs, IBM job IDs)
- Complete statistical summaries
- Meta-analysis across platforms
- Evidence hierarchy for manuscript
"""

import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy import stats
from collections import defaultdict
import re

# Paths
BASE_DIR = Path(r"c:\Users\Jack0\GitHub\Drift-Aware-Fault-Tolerance-QEC")
RESULTS_DIR = BASE_DIR / "results"
MULTI_PLATFORM = RESULTS_DIR / "multi_platform"
IBM_EXPERIMENTS = RESULTS_DIR / "ibm_experiments"
HARDWARE_VALIDATION = RESULTS_DIR / "hardware_validation"

def load_json(filepath):
    """Load JSON file."""
    try:
        with open(filepath, encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return None
            return json.loads(content)
    except Exception as e:
        return {"_error": str(e), "_file": str(filepath)}

def extract_ids_recursive(obj, ids_dict):
    """Recursively extract all task/job IDs from nested structures."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "task_id" and isinstance(value, str) and "arn:aws:braket" in value:
                ids_dict["braket_task_ids"].add(value)
            elif key == "job_id" and isinstance(value, str):
                ids_dict["ibm_job_ids"].add(value)
            elif key == "task_ids" and isinstance(value, list):
                for tid in value:
                    if isinstance(tid, str) and "arn:aws:braket" in tid:
                        ids_dict["braket_task_ids"].add(tid)
            elif key == "job_ids" and isinstance(value, list):
                for jid in value:
                    if isinstance(jid, str):
                        ids_dict["ibm_job_ids"].add(jid)
            else:
                extract_ids_recursive(value, ids_dict)
    elif isinstance(obj, list):
        for item in obj:
            extract_ids_recursive(item, ids_dict)

def analyze_single_file(filepath):
    """Deeply analyze a single JSON file."""
    data = load_json(filepath)
    if data is None:
        return None
    if "_error" in data:
        return {"file": filepath.name, "error": data["_error"]}
    
    result = {
        "file": filepath.name,
        "path": str(filepath),
        "size_bytes": filepath.stat().st_size,
    }
    
    # Extract IDs
    ids = {"braket_task_ids": set(), "ibm_job_ids": set()}
    extract_ids_recursive(data, ids)
    result["braket_task_ids"] = list(ids["braket_task_ids"])
    result["ibm_job_ids"] = list(ids["ibm_job_ids"])
    
    # Extract device/platform info
    for key in ["device", "backend", "platform", "device_arn", "device_key"]:
        if key in data:
            result[key] = data[key]
    
    # Extract experimental parameters
    for key in ["n_data_qubits", "shots", "n_rounds", "noise_level", "timestamp"]:
        if key in data:
            result[key] = data[key]
    
    # Extract results
    for key in ["raw_ler", "correctable_ler", "ler", "interaction", "interaction_effect"]:
        if key in data:
            result[key] = data[key]
    
    # Handle runs array
    if "runs" in data and isinstance(data["runs"], list):
        runs = data["runs"]
        result["n_runs"] = len(runs)
        
        # Extract interactions from runs
        interactions = []
        for run in runs:
            if isinstance(run, dict):
                if "interaction" in run:
                    interactions.append(run["interaction"])
                # Also extract task IDs from runs
                if "task_id" in run:
                    result["braket_task_ids"].append(run["task_id"])
        
        if interactions:
            result["interactions"] = interactions
            result["mean_interaction"] = float(np.mean(interactions))
            result["std_interaction"] = float(np.std(interactions, ddof=1)) if len(interactions) > 1 else 0
            
            if len(interactions) > 1:
                t_stat, p_two = stats.ttest_1samp(interactions, 0)
                p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
                result["p_value_one_tailed"] = float(p_one)
                result["t_statistic"] = float(t_stat)
                result["significant"] = p_one < 0.05
    
    # Handle experiments array (IonQ format)
    if "experiments" in data and isinstance(data["experiments"], list):
        experiments = data["experiments"]
        result["n_experiments"] = len(experiments)
        
        for exp in experiments:
            if isinstance(exp, dict) and "task_id" in exp:
                result["braket_task_ids"].append(exp["task_id"])
    
    # Handle comparison data
    if "comparison" in data:
        result["comparison"] = data["comparison"]
    
    # De-duplicate task IDs
    result["braket_task_ids"] = list(set(result["braket_task_ids"]))
    result["ibm_job_ids"] = list(set(result["ibm_job_ids"]))
    
    return result

def analyze_directory(directory, name):
    """Analyze all JSON files in a directory."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {name}")
    print(f"Directory: {directory}")
    print(f"{'='*80}")
    
    json_files = list(directory.glob("*.json"))
    jsonl_files = list(directory.glob("*.jsonl"))
    
    print(f"Found {len(json_files)} JSON files, {len(jsonl_files)} JSONL files")
    
    results = []
    all_braket_ids = set()
    all_ibm_ids = set()
    
    for f in sorted(json_files):
        print(f"\n  Analyzing: {f.name}")
        analysis = analyze_single_file(f)
        if analysis:
            results.append(analysis)
            all_braket_ids.update(analysis.get("braket_task_ids", []))
            all_ibm_ids.update(analysis.get("ibm_job_ids", []))
            
            # Print key findings
            if "mean_interaction" in analysis:
                print(f"    → Interaction: {analysis['mean_interaction']:.4f} (N={analysis.get('n_runs', 'N/A')})")
                if "p_value_one_tailed" in analysis:
                    sig = "SIGNIFICANT" if analysis.get("significant") else "not significant"
                    print(f"    → p-value: {analysis['p_value_one_tailed']:.4f} ({sig})")
            if "device" in analysis:
                print(f"    → Device: {analysis['device']}")
            if analysis.get("braket_task_ids"):
                print(f"    → Braket task IDs: {len(analysis['braket_task_ids'])}")
            if analysis.get("ibm_job_ids"):
                print(f"    → IBM job IDs: {len(analysis['ibm_job_ids'])}")
    
    # Handle JSONL files (IBM submitted jobs)
    for f in jsonl_files:
        print(f"\n  Analyzing JSONL: {f.name}")
        try:
            with open(f) as fp:
                lines = fp.readlines()
            for line in lines:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        if "job_id" in obj:
                            all_ibm_ids.add(obj["job_id"])
                    except:
                        pass
            print(f"    → Extracted job IDs from {len(lines)} lines")
        except Exception as e:
            print(f"    → Error: {e}")
    
    return {
        "name": name,
        "n_files": len(json_files),
        "results": results,
        "all_braket_ids": list(all_braket_ids),
        "all_ibm_ids": list(all_ibm_ids)
    }

def meta_analysis(all_results):
    """Perform meta-analysis across all experiments."""
    print(f"\n{'='*80}")
    print("META-ANALYSIS ACROSS ALL PLATFORMS")
    print(f"{'='*80}")
    
    # Collect all interaction effects with sample sizes
    studies = []
    
    for source in all_results:
        for result in source.get("results", []):
            if "mean_interaction" in result and "n_runs" in result:
                n = result["n_runs"]
                mean = result["mean_interaction"]
                std = result.get("std_interaction", 0.01)
                
                studies.append({
                    "file": result["file"],
                    "device": result.get("device", "Unknown"),
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "var": std**2 if std > 0 else 0.0001,
                    "p_value": result.get("p_value_one_tailed")
                })
    
    if not studies:
        print("No studies with interaction data found for meta-analysis")
        return None
    
    print(f"\nStudies included in meta-analysis: {len(studies)}")
    
    # Fixed-effects meta-analysis
    total_n = sum(s["n"] for s in studies)
    
    # Weighted mean (inverse variance weighting)
    weights = []
    for s in studies:
        w = s["n"] / s["var"] if s["var"] > 0 else s["n"]
        weights.append(w)
    
    total_weight = sum(weights)
    weighted_mean = sum(w * s["mean"] for w, s in zip(weights, studies)) / total_weight if total_weight > 0 else 0
    
    # Calculate pooled variance
    weighted_var = 1 / total_weight if total_weight > 0 else 0
    weighted_se = np.sqrt(weighted_var)
    
    # Z-test for meta-analysis
    z = weighted_mean / weighted_se if weighted_se > 0 else 0
    p_meta = 1 - stats.norm.cdf(abs(z)) if z < 0 else stats.norm.cdf(z)
    
    print(f"\n  Total N across all studies: {total_n}")
    print(f"  Weighted mean interaction: {weighted_mean:.4f}")
    print(f"  Weighted SE: {weighted_se:.4f}")
    print(f"  Z-statistic: {z:.3f}")
    print(f"  Meta-analysis p-value: {p_meta:.4f}")
    
    # Effect direction consistency
    n_negative = sum(1 for s in studies if s["mean"] < 0)
    print(f"\n  Studies with negative interaction: {n_negative}/{len(studies)} ({n_negative/len(studies)*100:.1f}%)")
    
    return {
        "n_studies": len(studies),
        "total_n": total_n,
        "weighted_mean": float(weighted_mean),
        "weighted_se": float(weighted_se),
        "z_statistic": float(z),
        "p_value": float(p_meta),
        "pct_negative": float(n_negative / len(studies) * 100),
        "studies": studies
    }

def generate_quantum_task_table(all_braket_ids, all_ibm_ids):
    """Generate table of quantum task IDs for SI."""
    print(f"\n{'='*80}")
    print("QUANTUM TASK ID INVENTORY")
    print(f"{'='*80}")
    
    # Parse Braket ARNs to extract device info
    braket_by_device = defaultdict(list)
    for arn in all_braket_ids:
        # Parse ARN: arn:aws:braket:region:account:quantum-task/task-id
        match = re.search(r'device/(qpu|simulator)/(\w+)/(\S+)', arn)
        if match:
            device_type, provider, device = match.groups()
            braket_by_device[f"{provider}/{device}"].append(arn)
        else:
            braket_by_device["other"].append(arn)
    
    print(f"\nAmazon Braket Task IDs by Device:")
    for device, arns in sorted(braket_by_device.items()):
        print(f"\n  {device}: {len(arns)} tasks")
        for arn in arns[:3]:  # Show first 3
            # Extract just the task ID
            task_id = arn.split("/")[-1]
            print(f"    - {task_id}")
        if len(arns) > 3:
            print(f"    ... and {len(arns) - 3} more")
    
    print(f"\nIBM Quantum Job IDs: {len(all_ibm_ids)}")
    for jid in list(all_ibm_ids)[:5]:
        print(f"  - {jid}")
    if len(all_ibm_ids) > 5:
        print(f"  ... and {len(all_ibm_ids) - 5} more")
    
    return {
        "braket_by_device": {k: v for k, v in braket_by_device.items()},
        "ibm_job_ids": list(all_ibm_ids)
    }

def main():
    print("="*80)
    print("DEEP COMPREHENSIVE EXPERIMENTAL ANALYSIS")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*80)
    
    all_results = []
    all_braket_ids = set()
    all_ibm_ids = set()
    
    # Analyze each directory
    for directory, name in [
        (MULTI_PLATFORM, "Multi-Platform Validation"),
        (HARDWARE_VALIDATION, "Hardware Validation"),
        (IBM_EXPERIMENTS, "IBM Experiments")
    ]:
        if directory.exists():
            result = analyze_directory(directory, name)
            all_results.append(result)
            all_braket_ids.update(result["all_braket_ids"])
            all_ibm_ids.update(result["all_ibm_ids"])
    
    # Perform meta-analysis
    meta = meta_analysis(all_results)
    
    # Generate task ID inventory
    task_inventory = generate_quantum_task_table(all_braket_ids, all_ibm_ids)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    total_files = sum(r["n_files"] for r in all_results)
    print(f"\nTotal JSON files analyzed: {total_files}")
    print(f"Total Braket task IDs: {len(all_braket_ids)}")
    print(f"Total IBM job IDs: {len(all_ibm_ids)}")
    
    if meta:
        print(f"\nMeta-analysis result:")
        print(f"  Weighted mean interaction: {meta['weighted_mean']:.4f}")
        print(f"  p-value: {meta['p_value']:.4f}")
        print(f"  {'SIGNIFICANT' if meta['p_value'] < 0.05 else 'Not significant'} at α = 0.05")
    
    # Compile complete output
    output = {
        "analysis_timestamp": datetime.now().isoformat(),
        "summary": {
            "total_files_analyzed": total_files,
            "total_braket_task_ids": len(all_braket_ids),
            "total_ibm_job_ids": len(all_ibm_ids)
        },
        "directories_analyzed": [r["name"] for r in all_results],
        "meta_analysis": meta,
        "quantum_task_ids": {
            "braket": list(all_braket_ids),
            "ibm": list(all_ibm_ids)
        },
        "detailed_results": all_results
    }
    
    # Save output
    output_file = RESULTS_DIR / "DEEP_EXPERIMENT_ANALYSIS.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n\nComplete analysis saved to: {output_file}")
    
    return output

if __name__ == "__main__":
    main()
