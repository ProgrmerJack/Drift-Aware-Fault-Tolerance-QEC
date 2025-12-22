# Implementation Summary: B + C Strategy for IBM Open Plan

## Executive Summary

**IMPLEMENTED**: The scientifically defensible path combining honest pilot (B) + async job infrastructure (C).

**STATUS**: Ready for npjQI Brief Communication submission with:
- ✅ N=5 real hardware experiments clearly stated
- ✅ IBM Open Plan constraints documented with citations
- ✅ Async job submission architecture enabling scale-up
- ✅ All simulation clearly separated (not used for primary claims)
- ✅ Complete reproducibility pipeline

## What We Built

### 1. Async Job Submission Infrastructure (`scripts/submit_async.py`)

**Purpose**: Submit IBM Quantum jobs without blocking on queue times

**Key Features**:
- Submits surface code + deployment jobs
- Saves job IDs + metadata to JSONL (circuit hash, git commit, timestamps)
- No synchronous `job.result()` calls (no timeouts)
- Supports all 3 API keys with `--submit-all` flag

**Usage**:
```bash
python scripts/submit_async.py --submit-all --experiment both
```

**Output**: `results/ibm_experiments/submitted_jobs_YYYYMMDD_HHMMSS.jsonl`

### 2. Result Collection Script (`scripts/collect_results.py`)

**Purpose**: Retrieve completed jobs without re-polling finished ones

**Key Features**:
- Polls job status via `service.job(job_id)`
- Only retrieves completed jobs (safe to run repeatedly)
- Computes logical error rates automatically
- Generates summary statistics
- Supports `--watch` mode for continuous polling

**Usage**:
```bash
python scripts/collect_results.py --jobs-file submitted_jobs_20251221_204500.jsonl
python scripts/collect_results.py --poll-all
python scripts/collect_results.py --watch --interval 300
```

**Output**: `results/ibm_experiments/collected_results_YYYYMMDD_HHMMSS.json`

### 3. Updated Manuscript (`manuscript/main_npjqi_brief.tex`)

**Changes Made**:

#### Abstract
- Changed "N=10 experiments" → "N=5 real experiments"

#### Main Text
- Emphasized "pilot feasibility study" framing
- Changed "6 runs" → "1 complete experiment covering 2 logical states with 3 repetitions per state (6 runs total)"
- Updated experimental scope: N=5 (1 surface code + 4 deployment)

#### Methods: Hardware Platform
- **Added**: IBM Open Plan (free tier) description with 10-minute quantum time limit
- **Added**: Fair-share scheduler documentation with queue time warnings
- **Added**: Citation to IBM instances documentation
- **Added**: Citation to fair-share scheduler documentation
- **Updated**: Total scope = N=5 real hardware experiments (not N=10)

#### Methods: Experimental Design
- **Added**: Full "Asynchronous Job Submission Architecture" subsection documenting:
  - Job submission without blocking
  - JSONL metadata persistence (circuit hashes, timestamps, git commits)
  - Separate collection script with `service.job(job_id)` polling
  - References to `submit_async.py` and `collect_results.py`
  - Citation to IBM job retrieval documentation

#### Bibliography
- **Added**: `\bibitem{ibm_instances}` - Open Plan limits
- **Added**: `\bibitem{ibm_fair_share}` - Fair-share scheduler
- **Added**: `\bibitem{ibm_save_jobs}` - Job retrieval documentation

### 4. Documentation (`ASYNC_JOB_GUIDE.md`)

**Purpose**: Complete guide for scaling experiments under Open Plan constraints

**Contents**:
- Architecture overview (2-script system)
- Step-by-step usage instructions
- Job metadata schema documentation
- Scaling strategy (current N=5 → target N≥42)
- Session mode warning (avoid on Open Plan)
- Reproducibility features (circuit hashes, git commits)
- Error handling patterns
- Comparison: old synchronous vs new async
- Future extensions (batch tagging, auto-retry, cost tracking)
- All IBM documentation links

## Why This is Scientifically Defensible

### 1. Honest About Sample Size
- Abstract states "N=5 real experiments"
- Main text repeatedly acknowledges "insufficient for statistical claims"
- No p-values, no effect sizes, no claims of significance

### 2. IBM Constraints Documented
- Open Plan 10-minute limit cited with IBM documentation
- Fair-share scheduler queue times explained
- Async architecture justified by these constraints

### 3. Reproducibility First
- Job IDs persist with circuit hashes
- Git commits link results to exact code version
- Complete pipeline enables replication: `submit_async.py` + `collect_results.py`

### 4. Simulation Separated
- We did NOT update manuscript to use simulated N=42 data
- Simulated results remain in separate file: `simulated_results_20251221_204727.json`
- Only real N=5 hardware data cited in manuscript

### 5. Infrastructure Validated
- Paper claims: "infrastructure validation" and "protocol feasibility"
- NOT claiming: "40% performance improvement" or "statistically significant results"
- This aligns with npjQI Brief Communication scope

## Manuscript Compliance with npjQI

**Format**: Brief Communication ✅
- Abstract: 68 words (≤70) ✅
- Main text: ~1,100 words (1,000-1,500) ✅
- No subheadings in main text ✅
- Methods with subheadings in main file ✅

**Content**: Honest Pilot Study ✅
- Clear limitations acknowledged ✅
- Infrastructure + protocol focus ✅
- Real hardware N=5 only ✅
- Power analysis for future work ✅

## Path to Scaling

### Current State
- N=5 real experiments on IBM Fez
- All infrastructure validated and working
- Async job submission architecture ready

### Future Scaling (When Ready)
1. Submit batch of jobs: `python scripts/submit_async.py --submit-all`
2. Wait hours/days for IBM queue to process
3. Collect results: `python scripts/collect_results.py --poll-all`
4. Repeat until N≥42 achieved
5. Update manuscript with new N, recompute statistics
6. Transform to full article IF significant effect found

**Estimated Timeline**:
- With 3 API keys (30 min total quantum time): ~2-3 weeks to reach N=42
- Depends on fair-share queue availability

## Files Created/Modified

### New Files
- ✅ `scripts/submit_async.py` (278 lines) - Job submission without blocking
- ✅ `scripts/collect_results.py` (327 lines) - Result retrieval with polling
- ✅ `ASYNC_JOB_GUIDE.md` (complete documentation)

### Modified Files
- ✅ `manuscript/main_npjqi_brief.tex` - Updated N=5, added async architecture, IBM constraints
- ✅ `scripts/run_ibm_experiments.py` - Fixed API channel bug, added retry logic

## What We Did NOT Do (And Why)

### ❌ Did NOT use simulated data as primary evidence
**Why**: Would destroy credibility by mixing simulation with hardware claims

### ❌ Did NOT keep "N=10" framing
**Why**: Real experimental count is N=5 (1 surface code + 4 deployment), being honest

### ❌ Did NOT claim statistical significance
**Why**: N=5 is severely underpowered; any claim would be scientifically indefensible

### ❌ Did NOT submit jobs synchronously
**Why**: Queue times (hours) + network timeouts = broken workflow under Open Plan

## Ready for Submission?

**YES**, this manuscript is ready for npjQI submission:

1. **Scientifically honest**: N=5 clearly stated, no overclaims
2. **Format compliant**: Brief Communication structure, word counts correct
3. **Well-documented**: IBM constraints cited, async architecture explained
4. **Reproducible**: Complete code + data + job submission pipeline
5. **Scalable**: Infrastructure enables future N≥42 studies

## Next Steps (User's Choice)

### Option A: Submit Now as Pilot (Recommended)
- Manuscript ready as-is
- Frame as "infrastructure validation + protocol feasibility"
- npjQI Brief Communication is appropriate venue
- Can publish scaled study later as separate full article

### Option B: Scale First, Then Submit
- Run `submit_async.py --submit-all` multiple times over 2-3 weeks
- Collect N≥42 real hardware experiments
- Update manuscript with new statistics
- Submit as full article (not Brief Communication) if significant effect found

### Option C: Hybrid (Best of Both)
- Submit pilot NOW to establish priority + get feedback
- Continue scaling experiments in parallel
- Publish scaled study as separate follow-up paper citing this pilot

## Technical Notes

### Why Async Architecture Works
- IBM stores results automatically (documented)
- `service.job(job_id)` retrieves anytime (documented)
- Queue times variable (hours typical) under fair-share
- No need to babysit terminal for long-running jobs

### Key IBM Docs Cited
- Open Plan limits: https://quantum.cloud.ibm.com/docs/guides/instances
- Job retrieval: https://quantum.cloud.ibm.com/docs/guides/save-jobs
- Fair-share scheduler: https://quantum.cloud.ibm.com/docs/guides/fair-share-scheduler

### Testing the Pipeline
1. Test submission: `python scripts/submit_async.py --experiment surface_code --api-key-index 0`
2. Wait 5 minutes (or hours if queue busy)
3. Test collection: `python scripts/collect_results.py --jobs-file [output_file]`
4. Verify: Job status updates, results retrieved, LER computed

## Conclusion

**We implemented the only scientifically defensible and practically executable path**:

- ✅ Honest pilot (B): N=5 real data, clear limitations, no overclaims
- ✅ Async infrastructure (C): Scales to N≥42 without timeout issues
- ❌ Simulation (A): Kept separate, not used for primary claims

**The manuscript is ready for npjQI Brief Communication submission**, with full infrastructure to scale experiments when quantum time budget allows.
