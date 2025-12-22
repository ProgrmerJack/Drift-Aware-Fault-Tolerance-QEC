#!/usr/bin/env python3
"""
collect_results.py - Retrieve Completed IBM Quantum Jobs
========================================================

Polls submitted jobs and retrieves completed results.
Safe to run repeatedly - only fetches new completions.

This script implements IBM's recommended job retrieval pattern for async workflows
where jobs may complete hours after submission due to queue times.

Usage:
    python scripts/collect_results.py --jobs-file submitted_jobs_20251221_123456.jsonl
    python scripts/collect_results.py --poll-all
    python scripts/collect_results.py --job-id d541c5onsj9s73b1d4lg

References:
- IBM job retrieval: https://quantum.cloud.ibm.com/docs/guides/save-jobs
- Job status lifecycle: https://quantum.cloud.ibm.com/docs/guides/monitor-job
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np

from qiskit_ibm_runtime import QiskitRuntimeService

# API Keys
API_KEYS = [
    "QXKvh5Ol-rQRrxbs2rdxlo9MLlCpNJioLB8p1_uujfkD",
    "2pbhDH38zmWHgFGw_7Pp8d1ugGvPaa5KR2aTMvW8LJfo",
    "wHH8qtEd9yjYFRRrBKNaExedCLE9JX9qiDG9w3krgYow",
]


def connect_to_ibm(api_key):
    """Connect to IBM Quantum."""
    channels = [
        {"channel": "ibm_cloud"},
        {"channel": "ibm_quantum_platform"},
    ]
    
    for channel_config in channels:
        try:
            service = QiskitRuntimeService(token=api_key, **channel_config)
            return service
        except Exception:
            continue
    
    raise RuntimeError("Failed to connect with any channel configuration")


def load_submitted_jobs(jobs_file):
    """Load job metadata from JSONL file."""
    jobs = []
    with open(jobs_file, "r") as f:
        for line in f:
            jobs.append(json.loads(line))
    return jobs


def compute_logical_error_rate(counts, logical_state='+'):
    """
    Compute logical error rate from measurement counts.
    
    For surface code:
    - Logical |+⟩: errors are odd-parity syndromes
    - Logical |0⟩: errors are any 1 in majority vote
    """
    total_shots = sum(counts.values())
    
    if logical_state == '+':
        # Count odd-parity outcomes (X-errors)
        errors = sum(count for bitstring, count in counts.items() 
                    if bin(int(bitstring, 2)).count('1') % 2 == 1)
    else:  # |0⟩
        # Count any non-zero outcomes
        errors = sum(count for bitstring, count in counts.items() 
                    if bitstring != '0' * len(bitstring))
    
    return errors / total_shots if total_shots > 0 else 0.0


def retrieve_job_result(service, job_metadata):
    """
    Retrieve result for a single job if completed.
    
    Returns:
        dict: Updated metadata with results, or None if not ready
    """
    job_id = job_metadata["job_id"]
    
    try:
        # Retrieve job object (does not wait)
        job = service.job(job_id)
        status = job.status()
        
        # Handle both enum and string status
        status_str = status.name if hasattr(status, 'name') else str(status)
        
        print(f"  Job {job_id}: {status_str}")
        
        if status_str in ["DONE", "COMPLETED"]:
            # Job finished successfully - get results
            result = job.result()
            
            # Extract counts from Sampler V2 result format
            pub_result = result[0]
            
            # New API: data is a DataBin with BitArray fields
            if hasattr(pub_result.data, 'c'):
                # Get BitArray and convert to counts
                bit_array = pub_result.data.c
                counts = bit_array.get_counts()
            elif hasattr(pub_result.data, 'final'):
                counts = pub_result.data.final.get_counts()
            elif hasattr(pub_result.data, 'meas'):
                counts = pub_result.data.meas.get_counts()
            else:
                # Fallback
                counts = pub_result.data.get_counts()
            
            # Compute error rate
            if job_metadata["experiment_type"] == "surface_code":
                logical_state = job_metadata.get("logical_state", "+")
                ler = compute_logical_error_rate(counts, logical_state)
            else:  # deployment
                # For repetition code: count majority errors
                total = sum(counts.values())
                all_zeros = counts.get("0" * 5, 0)
                ler = 1.0 - (all_zeros / total)
            
            # Update metadata
            job_metadata["status"] = "completed"
            job_metadata["completion_time"] = datetime.now().isoformat()
            job_metadata["logical_error_rate"] = round(ler, 6)
            job_metadata["raw_counts"] = dict(counts)
            job_metadata["total_shots"] = sum(counts.values())
            
            return job_metadata
        
        elif status_str in ["CANCELLED", "FAILED", "ERROR"]:
            # Job failed
            job_metadata["status"] = status_str.lower()
            job_metadata["completion_time"] = datetime.now().isoformat()
            job_metadata["error"] = str(job.error_message()) if hasattr(job, 'error_message') else "Unknown error"
            
            return job_metadata
        
        else:
            # Still queued or running
            job_metadata["status"] = status_str.lower()
            job_metadata["last_check"] = datetime.now().isoformat()
            
            return None  # Not ready yet
    
    except Exception as e:
        print(f"    Error retrieving job: {e}")
        job_metadata["status"] = "retrieval_error"
        job_metadata["error"] = str(e)
        return None


def poll_jobs(jobs_file, max_retries=3, wait_between_checks=30):
    """
    Poll all jobs in file and retrieve completed ones.
    
    Args:
        jobs_file: Path to submitted_jobs_*.jsonl
        max_retries: Max connection retries per job
        wait_between_checks: Seconds to wait between polling cycles
    """
    output_dir = Path(jobs_file).parent
    
    # Load jobs
    jobs = load_submitted_jobs(jobs_file)
    print(f"\nLoaded {len(jobs)} jobs from {jobs_file}")
    
    # Group by API key for connection efficiency
    jobs_by_key = {}
    for job in jobs:
        key_idx = job["api_key_index"]
        if key_idx not in jobs_by_key:
            jobs_by_key[key_idx] = []
        jobs_by_key[key_idx].append(job)
    
    completed_jobs = []
    pending_jobs = []
    
    print("\n" + "=" * 70)
    print("POLLING JOB STATUS")
    print("=" * 70)
    
    for key_idx, key_jobs in jobs_by_key.items():
        print(f"\nAPI Key {key_idx + 1} ({len(key_jobs)} jobs):")
        
        try:
            service = connect_to_ibm(API_KEYS[key_idx])
            
            for job_meta in key_jobs:
                # Skip if already completed in previous run
                if job_meta.get("status") == "completed":
                    completed_jobs.append(job_meta)
                    continue
                
                # Try to retrieve result
                updated_meta = retrieve_job_result(service, job_meta)
                
                if updated_meta and updated_meta["status"] == "completed":
                    completed_jobs.append(updated_meta)
                elif updated_meta and updated_meta["status"] in ["failed", "cancelled", "error"]:
                    print(f"    ✗ Job failed: {updated_meta.get('error', 'Unknown')}")
                    completed_jobs.append(updated_meta)  # Still save it
                else:
                    # Still pending
                    pending_jobs.append(job_meta)
        
        except Exception as e:
            print(f"  ✗ Connection error for API key {key_idx}: {e}")
            # Keep all jobs as pending
            pending_jobs.extend([j for j in key_jobs if j.get("status") != "completed"])
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Update original file with status
    with open(jobs_file, "w") as f:
        for job in completed_jobs + pending_jobs:
            f.write(json.dumps(job) + "\n")
    
    # Save completed results separately
    if completed_jobs:
        results_file = output_dir / f"collected_results_{timestamp}.json"
        
        with open(results_file, "w") as f:
            json.dump({
                "collection_time": datetime.now().isoformat(),
                "total_jobs": len(jobs),
                "completed": len(completed_jobs),
                "pending": len(pending_jobs),
                "results": completed_jobs,
            }, f, indent=2)
        
        print(f"\n{'=' * 70}")
        print(f"COLLECTION SUMMARY")
        print(f"{'=' * 70}")
        print(f"Completed: {len(completed_jobs)}/{len(jobs)}")
        print(f"Pending:   {len(pending_jobs)}/{len(jobs)}")
        print(f"\nResults saved to: {results_file}")
        
        # Print summary statistics
        if any(j["status"] == "completed" for j in completed_jobs):
            print(f"\n{'=' * 70}")
            print("PRELIMINARY RESULTS")
            print("=" * 70)
            
            surface_code_jobs = [j for j in completed_jobs 
                                if j["experiment_type"] == "surface_code" and j["status"] == "completed"]
            deployment_jobs = [j for j in completed_jobs 
                              if j["experiment_type"] == "deployment" and j["status"] == "completed"]
            
            if surface_code_jobs:
                lers = [j["logical_error_rate"] for j in surface_code_jobs]
                print(f"\nSurface Code (N={len(surface_code_jobs)}):")
                print(f"  Mean LER: {np.mean(lers):.4f} ± {np.std(lers):.4f}")
                print(f"  Range: [{np.min(lers):.4f}, {np.max(lers):.4f}]")
            
            if deployment_jobs:
                baseline = [j["logical_error_rate"] for j in deployment_jobs if j["mode"] == "baseline"]
                daqec = [j["logical_error_rate"] for j in deployment_jobs if j["mode"] == "daqec"]
                
                print(f"\nDeployment:")
                if baseline:
                    print(f"  Baseline (N={len(baseline)}): {np.mean(baseline):.4f} ± {np.std(baseline):.4f}")
                if daqec:
                    print(f"  DAQEC (N={len(daqec)}):    {np.mean(daqec):.4f} ± {np.std(daqec):.4f}")
                if baseline and daqec:
                    improvement = (1 - np.mean(daqec) / np.mean(baseline)) * 100
                    print(f"  Improvement: {improvement:.1f}%")
    
    else:
        print(f"\nNo completed jobs yet. {len(pending_jobs)} still pending.")
        print(f"Run this script again later to check for completions.")


def main():
    parser = argparse.ArgumentParser(
        description="Collect completed IBM Quantum job results"
    )
    parser.add_argument(
        "--jobs-file",
        type=Path,
        help="Path to submitted_jobs_*.jsonl file",
    )
    parser.add_argument(
        "--poll-all",
        action="store_true",
        help="Poll all job files in results/ibm_experiments/",
    )
    parser.add_argument(
        "--job-id",
        help="Retrieve specific job by ID",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Poll continuously until all jobs complete",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between polling cycles (default: 300)",
    )
    
    args = parser.parse_args()
    
    if args.job_id:
        # Single job retrieval
        print(f"Retrieving job {args.job_id}...")
        # Try all API keys
        for key_idx, api_key in enumerate(API_KEYS):
            try:
                service = connect_to_ibm(api_key)
                job = service.job(args.job_id)
                status = job.status()
                status_str = status.name if hasattr(status, 'name') else str(status)
                
                print(f"Found with API key {key_idx + 1}")
                print(f"Status: {status_str}")
                
                if status_str in ["DONE", "COMPLETED"]:
                    result = job.result()
                    print(f"Result: {result}")
                break
            except Exception as e:
                continue
    
    elif args.poll_all:
        # Poll all job files
        results_dir = Path(__file__).parent.parent / "results" / "ibm_experiments"
        job_files = list(results_dir.glob("submitted_jobs_*.jsonl"))
        
        if not job_files:
            print("No submitted job files found!")
            return
        
        for job_file in job_files:
            print(f"\nProcessing {job_file.name}...")
            poll_jobs(job_file)
    
    elif args.jobs_file:
        # Poll specific file
        if args.watch:
            # Continuous polling
            while True:
                poll_jobs(args.jobs_file)
                
                # Check if all done
                jobs = load_submitted_jobs(args.jobs_file)
                pending = sum(1 for j in jobs if j.get("status") not in ["completed", "failed", "cancelled"])
                
                if pending == 0:
                    print("\nAll jobs completed!")
                    break
                
                print(f"\n{pending} jobs still pending. Waiting {args.interval}s...")
                time.sleep(args.interval)
        else:
            # Single poll
            poll_jobs(args.jobs_file)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
