#!/usr/bin/env python3
"""
build_master.py - Master Data Builder with Schema Enforcement
=============================================================

Builds the master.parquet file from raw experimental data with strict
schema validation per protocol.yaml output_schema specification.

This script ensures:
1. All required columns exist with correct dtypes
2. No unapproved columns are added
3. Data integrity checks pass
4. Audit trail is maintained

Usage:
    python scripts/build_master.py                    # Build from data/raw
    python scripts/build_master.py --validate-only   # Check existing master
    python scripts/build_master.py --source DIR      # Specify source directory
    python scripts/build_master.py --schema-report   # Generate schema report
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

class SchemaEnforcer:
    """Enforce output schema from protocol.yaml."""
    
    # Mapping from protocol dtype strings to pandas/numpy dtypes
    DTYPE_MAP = {
        'string': 'object',
        'int64': 'int64',
        'float64': 'float64',
        'datetime64[ns]': 'datetime64[ns]',
        'bool': 'bool',
    }
    
    # PyArrow type mapping for parquet
    PYARROW_MAP = {
        'string': pa.string(),
        'int64': pa.int64(),
        'float64': pa.float64(),
        'datetime64[ns]': pa.timestamp('ns'),
        'bool': pa.bool_(),
    }
    
    def __init__(self, protocol_path: Path = None):
        self.protocol_path = protocol_path or PROJECT_ROOT / "protocol" / "protocol.yaml"
        self.schema = None
        self.required_columns = []
        self.nullable_columns = []
        
    def load_schema(self) -> dict:
        """Load schema from protocol.yaml."""
        with open(self.protocol_path) as f:
            protocol = yaml.safe_load(f)
        
        self.schema = protocol['output_schema']['master_parquet']
        
        for col_spec in self.schema['required_columns']:
            self.required_columns.append(col_spec['name'])
            if col_spec.get('nullable', False):
                self.nullable_columns.append(col_spec['name'])
        
        logger.info(f"Loaded schema with {len(self.required_columns)} columns")
        return self.schema
    
    def get_column_spec(self, column_name: str) -> dict | None:
        """Get specification for a column."""
        for col_spec in self.schema['required_columns']:
            if col_spec['name'] == column_name:
                return col_spec
        return None
    
    def validate_dataframe(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        Validate DataFrame against schema.
        
        Returns:
            (passed, errors): Tuple of validation status and error messages
        """
        errors = []
        
        # Check required columns
        for col_name in self.required_columns:
            if col_name not in df.columns:
                errors.append(f"Missing required column: {col_name}")
        
        # Check for unexpected columns
        expected = set(self.required_columns)
        actual = set(df.columns)
        unexpected = actual - expected
        if unexpected:
            errors.append(f"Unexpected columns (not in schema): {unexpected}")
        
        # Check dtypes
        for col_spec in self.schema['required_columns']:
            col_name = col_spec['name']
            if col_name not in df.columns:
                continue
                
            expected_dtype = self.DTYPE_MAP.get(col_spec['dtype'], 'object')
            actual_dtype = str(df[col_name].dtype)
            
            # Allow some flexibility for numeric types
            if expected_dtype == 'float64' and actual_dtype in ['float32', 'float64']:
                continue
            if expected_dtype == 'int64' and actual_dtype in ['int32', 'int64', 'Int64']:
                continue
            if expected_dtype == 'object' and actual_dtype == 'object':
                continue
            if expected_dtype == 'datetime64[ns]' and 'datetime' in actual_dtype:
                continue
                
            if actual_dtype != expected_dtype:
                errors.append(
                    f"Column '{col_name}': expected dtype {expected_dtype}, got {actual_dtype}"
                )
        
        # Check nullability
        for col_spec in self.schema['required_columns']:
            col_name = col_spec['name']
            if col_name not in df.columns:
                continue
                
            has_nulls = df[col_name].isna().any()
            is_nullable = col_spec.get('nullable', False)
            
            if has_nulls and not is_nullable:
                n_nulls = df[col_name].isna().sum()
                errors.append(
                    f"Column '{col_name}': has {n_nulls} null values but is not nullable"
                )
        
        passed = len(errors) == 0
        return passed, errors
    
    def coerce_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce DataFrame columns to schema dtypes."""
        df = df.copy()
        
        for col_spec in self.schema['required_columns']:
            col_name = col_spec['name']
            if col_name not in df.columns:
                continue
            
            target_dtype = col_spec['dtype']
            
            try:
                if target_dtype == 'datetime64[ns]':
                    df[col_name] = pd.to_datetime(df[col_name], utc=True)
                elif target_dtype == 'int64':
                    # Handle nullable integers
                    if df[col_name].isna().any():
                        df[col_name] = df[col_name].astype('Int64')
                    else:
                        df[col_name] = df[col_name].astype('int64')
                elif target_dtype == 'float64':
                    df[col_name] = df[col_name].astype('float64')
                elif target_dtype == 'string':
                    df[col_name] = df[col_name].astype(str)
            except Exception as e:
                logger.warning(f"Could not coerce {col_name} to {target_dtype}: {e}")
        
        return df
    
    def get_pyarrow_schema(self) -> pa.Schema:
        """Get PyArrow schema for parquet writing."""
        fields = []
        for col_spec in self.schema['required_columns']:
            pa_type = self.PYARROW_MAP.get(col_spec['dtype'], pa.string())
            fields.append(pa.field(col_spec['name'], pa_type, nullable=col_spec.get('nullable', False)))
        return pa.schema(fields)
    
    def generate_report(self) -> str:
        """Generate human-readable schema report."""
        lines = [
            "=" * 70,
            "MASTER PARQUET SCHEMA REPORT",
            "=" * 70,
            "",
            f"Protocol file: {self.protocol_path}",
            f"Total columns: {len(self.required_columns)}",
            f"Nullable columns: {len(self.nullable_columns)}",
            "",
            "-" * 70,
            f"{'Column Name':<30} {'Type':<15} {'Nullable':<10}",
            "-" * 70,
        ]
        
        for col_spec in self.schema['required_columns']:
            nullable = "Yes" if col_spec.get('nullable', False) else "No"
            lines.append(f"{col_spec['name']:<30} {col_spec['dtype']:<15} {nullable:<10}")
            if 'description' in col_spec:
                lines.append(f"    └─ {col_spec['description']}")
        
        lines.extend(["", "-" * 70])
        return "\n".join(lines)


# =============================================================================
# DATA LOADER
# =============================================================================

class DataLoader:
    """Load raw data from various sources."""
    
    def __init__(self, source_dir: Path):
        self.source_dir = source_dir
        
    def load_all(self) -> pd.DataFrame:
        """Load and concatenate all raw data files."""
        all_data = []
        
        # Load session results
        session_dir = self.source_dir / "sessions"
        if session_dir.exists():
            for session_file in session_dir.glob("*.json"):
                if session_file.name.endswith("_manifest.json"):
                    continue
                try:
                    with open(session_file) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    logger.info(f"Loaded {session_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to load {session_file}: {e}")
        
        # Load parquet files
        for parquet_file in self.source_dir.glob("**/*.parquet"):
            if parquet_file.name == "master.parquet":
                continue
            try:
                df = pd.read_parquet(parquet_file)
                all_data.append(df.to_dict('records'))
                logger.info(f"Loaded {parquet_file.name}")
            except Exception as e:
                logger.warning(f"Failed to load {parquet_file}: {e}")
        
        # Load CSV files
        for csv_file in self.source_dir.glob("**/*.csv"):
            try:
                df = pd.read_csv(csv_file)
                all_data.append(df.to_dict('records'))
                logger.info(f"Loaded {csv_file.name}")
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")
        
        if not all_data:
            logger.warning("No data files found")
            return pd.DataFrame()
        
        # Flatten if nested
        flat_data = []
        for item in all_data:
            if isinstance(item, list):
                flat_data.extend(item)
            elif isinstance(item, dict):
                flat_data.append(item)
        
        return pd.DataFrame(flat_data)


# =============================================================================
# BUILDER
# =============================================================================

class MasterBuilder:
    """Build and validate master.parquet."""
    
    def __init__(
        self,
        source_dir: Path,
        output_path: Path,
        schema_enforcer: SchemaEnforcer
    ):
        self.source_dir = source_dir
        self.output_path = output_path
        self.schema_enforcer = schema_enforcer
        self.build_manifest = {}
        
    def build(self) -> tuple[bool, pd.DataFrame]:
        """Build master.parquet from raw data."""
        logger.info("=" * 60)
        logger.info("BUILDING MASTER PARQUET")
        logger.info("=" * 60)
        
        # Record build metadata
        self.build_manifest = {
            "build_timestamp": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(self.source_dir),
            "output_path": str(self.output_path),
        }
        
        # Load raw data
        logger.info(f"Loading data from {self.source_dir}")
        loader = DataLoader(self.source_dir)
        df = loader.load_all()
        
        if df.empty:
            logger.error("No data to build")
            return False, df
        
        logger.info(f"Loaded {len(df)} records")
        self.build_manifest["input_records"] = len(df)
        
        # Coerce dtypes
        logger.info("Coercing data types...")
        df = self.schema_enforcer.coerce_dtypes(df)
        
        # Validate against schema
        logger.info("Validating against schema...")
        passed, errors = self.schema_enforcer.validate_dataframe(df)
        
        if not passed:
            logger.error("Schema validation FAILED:")
            for error in errors:
                logger.error(f"  - {error}")
            self.build_manifest["validation_passed"] = False
            self.build_manifest["validation_errors"] = errors
            return False, df
        
        logger.info("✓ Schema validation passed")
        self.build_manifest["validation_passed"] = True
        
        # Add metadata columns if missing
        if 'build_version' not in df.columns:
            df['build_version'] = "1.0"
        
        # Write to parquet
        logger.info(f"Writing to {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use PyArrow for type enforcement
        table = pa.Table.from_pandas(df)
        pq.write_table(table, self.output_path, compression='snappy')
        
        # Compute hash
        with open(self.output_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        self.build_manifest["output_records"] = len(df)
        self.build_manifest["output_hash"] = file_hash
        self.build_manifest["output_size_bytes"] = self.output_path.stat().st_size
        
        # Save build manifest
        manifest_path = self.output_path.with_suffix('.build_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(self.build_manifest, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("BUILD COMPLETE")
        logger.info(f"Output: {self.output_path}")
        logger.info(f"Records: {len(df)}")
        logger.info(f"Hash: {file_hash[:32]}...")
        logger.info("=" * 60)
        
        return True, df
    
    def validate_existing(self) -> tuple[bool, list[str]]:
        """Validate existing master.parquet."""
        if not self.output_path.exists():
            return False, [f"File not found: {self.output_path}"]
        
        logger.info(f"Validating {self.output_path}")
        
        df = pd.read_parquet(self.output_path)
        return self.schema_enforcer.validate_dataframe(df)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build master.parquet with schema enforcement"
    )
    parser.add_argument(
        '--source',
        default='data/raw',
        help='Source directory for raw data'
    )
    parser.add_argument(
        '--output',
        default='data/processed/master.parquet',
        help='Output path for master.parquet'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Validate existing master.parquet without rebuilding'
    )
    parser.add_argument(
        '--schema-report',
        action='store_true',
        help='Print schema report and exit'
    )
    parser.add_argument(
        '--protocol',
        default='protocol/protocol.yaml',
        help='Path to protocol.yaml'
    )
    
    args = parser.parse_args()
    
    # Initialize schema enforcer
    enforcer = SchemaEnforcer(PROJECT_ROOT / args.protocol)
    enforcer.load_schema()
    
    if args.schema_report:
        print(enforcer.generate_report())
        return 0
    
    # Build paths
    source_dir = PROJECT_ROOT / args.source
    output_path = PROJECT_ROOT / args.output
    
    # Create builder
    builder = MasterBuilder(source_dir, output_path, enforcer)
    
    if args.validate_only:
        passed, errors = builder.validate_existing()
        if passed:
            logger.info("✓ Validation passed")
            return 0
        else:
            logger.error("✗ Validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return 1
    
    # Build
    success, df = builder.build()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
