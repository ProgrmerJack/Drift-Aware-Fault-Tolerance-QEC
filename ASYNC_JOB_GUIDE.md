# Async Job Submission Guide for IBM Quantum Open Plan

## Overview

This guide documents the asynchronous job submission architecture for scaling experiments under IBM Quantum Open Plan constraints.

**Key Constraint**: Open Plan accounts have ~10 minutes quantum time per 28-day window ([IBM Docs](https://quantum.cloud.ibm.com/docs/guides/instances))

**Key Solution**: Submit jobs without waiting → save job IDs → retrieve results later

## Architecture

### Two-Script System

1. **`submit_async.py`** - Submits jobs, saves metadata, exits immediately
2. **`collect_results.py`** - Polls job status, retrieves completed results

### Why This Works

IBM stores all job results automatically. You can retrieve them anytime using the job ID, even hours/days later ([IBM Docs](https://quantum.cloud.ibm.com/docs/guides/save-jobs)).

Queue times can be long (hours) under fair-share scheduler ([IBM Docs](https://quantum.cloud.ibm.com/docs/guides/fair-share-scheduler)), so synchronous waiting wastes your terminal session and risks network timeouts.

## Usage

### Step 1: Submit Jobs

```bash
# Submit with single API key
python scripts/submit_async.py --experiment both --api-key-index 0

# Submit with all 3 API keys
python scripts/submit_async.py --submit-all --experiment both

# Submit only surface code
python scripts/submit_async.py --experiment surface_code --api-key-index 1

# Submit only deployment
python scripts/submit_async.py --experiment deployment --api-key-index 2
```

**Output**: Creates `results/ibm_experiments/submitted_jobs_YYYYMMDD_HHMMSS.jsonl`

### Step 2: Wait (optional)

Jobs are now in IBM's queue. You can:
- Close your terminal
- Shut down your computer
- Check status later when convenient

Typical queue times: 10 minutes to several hours depending on load.

### Step 3: Retrieve Results

```bash
# Poll specific job file once
python scripts/collect_results.py --jobs-file submitted_jobs_20251221_204500.jsonl

# Poll all job files
python scripts/collect_results.py --poll-all

# Continuous polling until all complete
python scripts/collect_results.py --jobs-file submitted_jobs_20251221_204500.jsonl --watch --interval 300

# Check specific job by ID
python scripts/collect_results.py --job-id d541c5onsj9s73b1d4lg
```

**Output**: Creates `results/ibm_experiments/collected_results_YYYYMMDD_HHMMSS.json`

## Job Metadata Schema

Each line in `submitted_jobs_*.jsonl` contains:

```json
{
  "job_id": "d541c5onsj9s73b1d4lg",
  "experiment_type": "surface_code",
  "backend": "ibm_fez",
  "logical_state": "+",
  "shots": 4096,
  "circuit_hash": "a3f5b2c1d4e6f7g8",
  "transpiled_depth": 409,
  "transpiled_gates": 1170,
  "submission_time": "2025-12-21T20:45:15.675969",
  "api_key_index": 0,
  "git_commit": "abc123def456",
  "status": "submitted"
}
```

After collection, status updates to: `completed`, `failed`, `cancelled`, or `error`.

## Scaling Strategy

### Current Pilot (N=5)

- 1 surface code experiment (1 submission × 6 runs)
- 4 deployment sessions (4 submissions)
- **Total: 5 IBM jobs**

### Scaled Study (N≥42 for 80% power)

**Target**: 42 deployment sessions + 12 surface code experiments

**With 3 API keys (30 min total quantum time)**:
- Each surface code run: ~30 seconds quantum time
- Each deployment session: ~20 seconds quantum time
- **Feasible**: Yes, within time budget

**Execution plan**:
```bash
# Day 1: Submit batch 1
python scripts/submit_async.py --submit-all --experiment both

# Day 2-7: Wait for jobs to complete
python scripts/collect_results.py --poll-all

# Day 8: Submit batch 2 (repeat until N≥42)
python scripts/submit_async.py --submit-all --experiment deployment
```

### Avoiding Session Mode

**Do NOT use** IBM Session mode on Open Plan. Sessions charge for wall-clock time while QPU waits ([IBM Docs](https://quantum.cloud.ibm.com/docs/guides/execution-modes-faq)), burning your 10-minute budget.

Use plain job submission (as implemented in `submit_async.py`).

### Setting Execution Time Limits

Each job in `submit_async.py` can set `max_execution_time` to cap quantum time consumption ([IBM Docs](https://quantum.cloud.ibm.com/docs/guides/max-execution-time)):

```python
sampler = Sampler(mode=backend)
job = sampler.run(
    [isa_circuit],
    shots=4096,
    options={"max_execution_time": 60}  # 60 seconds max
)
```

## Monitoring Progress

### Via collect_results.py

```bash
# Check status (no downloads if jobs still queued)
python scripts/collect_results.py --jobs-file submitted_jobs_20251221_204500.jsonl
```

Prints:
```
Job d541c5onsj9s73b1d4lg: QUEUED
Job d541c6onsj9s73b1d4mh: RUNNING
Job d541c7onsj9s73b1d4ni: DONE
  ✓ Completed: 1/3
  Pending: 2/3
```

### Via IBM Quantum Web UI

1. Go to https://quantum.cloud.ibm.com/workloads
2. Find your jobs by ID
3. View status, queue position, results

## Reproducibility

All job metadata includes:
- **Circuit hash**: Detect accidental circuit changes
- **Git commit**: Link results to exact code version
- **Backend snapshot**: Calibration data at submission time
- **Submission timestamp**: Track temporal drift

This enables:
- Deduplication (don't rerun identical circuits)
- Provenance tracking (manuscript → job ID → code → data)
- Temporal analysis (drift correlations)

## Error Handling

### Failed Jobs

`collect_results.py` marks jobs as `failed` and saves error message:

```json
{
  "job_id": "d541c5onsj9s73b1d4lg",
  "status": "failed",
  "error": "Backend went offline during execution",
  "completion_time": "2025-12-21T22:15:30.123456"
}
```

### Network Timeouts

If `collect_results.py` crashes, just run it again. It:
- Reads existing status from JSONL
- Skips already-completed jobs
- Only polls pending jobs

**Safe to run repeatedly.**

### Lost Job IDs

If you lose the JSONL file, you can still retrieve jobs:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel="ibm_cloud", token="YOUR_TOKEN")

# List recent jobs
jobs = service.jobs(limit=50, pending=False)
for job in jobs:
    print(f"{job.job_id()}: {job.status()}")
```

## Comparison to Old Synchronous Approach

### ❌ Old (Broken)

```python
job = sampler.run([circuit], shots=4096)
result = job.result()  # BLOCKS FOR HOURS, TIMES OUT
```

**Problems**:
- Waits for queue (hours)
- Network timeouts kill process
- Loses all progress if crash
- Can't scale beyond 1-2 jobs

### ✅ New (Async)

```python
job = sampler.run([circuit], shots=4096)
job_id = job.job_id()
save_metadata(job_id)  # Persist immediately
# Exit, come back later
```

**Benefits**:
- Submit all jobs in 5 minutes
- Results retrieve anytime
- Network issues = just retry
- Scales to 100+ jobs

## Future Extensions

### Batch Submission with Job Tags

IBM supports job tagging for organization:

```python
job = sampler.run(
    [circuit],
    job_tags=["surface_code", "pilot_study", "batch_001"]
)
```

Then retrieve by tag:

```python
jobs = service.jobs(job_tags=["pilot_study"])
```

### Automatic Retry on Failure

Extend `collect_results.py` to auto-resubmit failed jobs:

```python
if job_meta["status"] == "failed":
    if job_meta.get("retry_count", 0) < 3:
        new_job = resubmit_circuit(job_meta)
        job_meta["retry_count"] += 1
        job_meta["job_id"] = new_job.job_id()
```

### Cost Tracking

Track quantum time usage per session:

```python
job_result = job.result()
usage = job.usage_estimation  # Quantum seconds
total_used += usage
remaining = (10 * 60) - total_used  # 10 min budget
```

## References

- [IBM Quantum Open Plan limits](https://quantum.cloud.ibm.com/docs/guides/instances)
- [IBM job retrieval docs](https://quantum.cloud.ibm.com/docs/guides/save-jobs)
- [Fair-share scheduler](https://quantum.cloud.ibm.com/docs/guides/fair-share-scheduler)
- [Max execution time](https://quantum.cloud.ibm.com/docs/guides/max-execution-time)
- [Execution modes FAQ](https://quantum.cloud.ibm.com/docs/guides/execution-modes-faq)

## Questions?

See `scripts/submit_async.py` and `scripts/collect_results.py` for implementation details.
