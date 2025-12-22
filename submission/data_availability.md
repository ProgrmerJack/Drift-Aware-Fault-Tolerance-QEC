# Data Availability Statement

> This statement will be included verbatim in the manuscript.
> Update the Zenodo DOI after data deposit.

---

## Data Availability

All data supporting the findings of this study are publicly available.

### Primary Data Deposit

The complete dataset is deposited at Zenodo under DOI: **[10.5281/zenodo.XXXXXXX]** (placeholder - update after deposit).

The deposit contains:

1. **Raw data** (`data/raw/`)
   - Syndrome measurement outcomes (bitstrings)
   - Probe circuit measurement outcomes
   - Backend calibration snapshots
   - Session metadata and timestamps

2. **Processed data** (`data/processed/`)
   - `master.parquet`: Consolidated analysis dataset with all derived metrics
   - Session-level aggregates
   - Statistical test results

3. **Source data** (`source_data/`)
   - `SourceData.xlsx`: Figure source data per Nature policy
   - Individual CSV files for each figure panel

### Data Format Specifications

- **Parquet files**: Apache Parquet format, readable with pandas, pyarrow, or polars
- **Excel files**: Microsoft Excel format (.xlsx), compatible with Excel 2010+
- **JSON files**: UTF-8 encoded, human-readable configuration and metadata
- **CSV files**: UTF-8 encoded, comma-separated, with header row

### Data Codebook

A complete data codebook (`DATA_CODEBOOK.md`) is included in the repository, documenting:
- Variable names and definitions
- Units and valid ranges
- Missing value conventions
- Derived variable formulas

### IBM Quantum Backend Data

Raw calibration data from IBM Quantum backends is accessed via the IBM Quantum API. Historical calibration data for the backends used in this study (ibm_brisbane, ibm_kyoto, ibm_osaka) is included in the deposit. Current calibration data is accessible at https://quantum.ibm.com/ with a free IBM Quantum account.

### Data Access

No restrictions apply to data access. All data are released under Creative Commons Attribution 4.0 International (CC BY 4.0) license.

### Reproducibility

To reproduce all analyses from the deposited data:

```bash
# Clone repository
git clone https://github.com/[USER]/Drift-Aware-Fault-Tolerance-QEC.git
cd Drift-Aware-Fault-Tolerance-QEC

# Download data
wget https://zenodo.org/record/XXXXXXX/files/data.zip
unzip data.zip -d data/

# Install dependencies
pip install -r requirements.txt

# Run analysis
python protocol/run_protocol.py --mode=analysis
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | [DATE] | Initial deposit with manuscript submission |

---

## Contact

For data access questions, contact the corresponding author at [EMAIL].
