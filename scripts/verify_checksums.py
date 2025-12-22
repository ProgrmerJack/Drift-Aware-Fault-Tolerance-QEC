#!/usr/bin/env python3
"""
verify_checksums.py - Verify Benchmark Data Integrity
======================================================

Verifies SHA-256 checksums of all benchmark data files against
published values. This is Step 2 of the independent reproduction protocol.

Usage:
    python scripts/verify_checksums.py
    python scripts/verify_checksums.py --verbose
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKSUM_FILE = DATA_DIR / "checksums.json"

# Files to verify
CRITICAL_FILES = [
    "master.parquet",
    "SourceData.xlsx",
]

OPTIONAL_FILES = [
    "calibration_snapshots.parquet",
    "syndrome_data.tar.gz",
]


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_expected_checksums() -> dict:
    """Load expected checksums from JSON file."""
    if not CHECKSUM_FILE.exists():
        print(f"Warning: {CHECKSUM_FILE} not found")
        print("Creating template checksums.json from current files...")
        return {}
    
    with open(CHECKSUM_FILE) as f:
        return json.load(f)


def generate_checksums() -> dict:
    """Generate checksums for all data files."""
    checksums = {}
    for filename in CRITICAL_FILES + OPTIONAL_FILES:
        filepath = DATA_DIR / filename
        if filepath.exists():
            checksums[filename] = compute_sha256(filepath)
    return checksums


def main():
    parser = argparse.ArgumentParser(description="Verify benchmark data checksums")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full hashes")
    parser.add_argument("--generate", action="store_true", help="Generate checksums.json")
    args = parser.parse_args()

    print("=" * 60)
    print("DAQEC Benchmark Checksum Verification")
    print("=" * 60)
    print()

    if args.generate:
        checksums = generate_checksums()
        with open(CHECKSUM_FILE, "w") as f:
            json.dump(checksums, f, indent=2)
        print(f"✓ Generated {CHECKSUM_FILE}")
        for filename, checksum in checksums.items():
            print(f"  {filename}: {checksum[:16]}...")
        sys.exit(0)

    expected = load_expected_checksums()
    
    all_verified = True
    missing = []
    mismatched = []
    verified = []

    for filename in CRITICAL_FILES:
        filepath = DATA_DIR / filename
        
        if not filepath.exists():
            print(f"✗ {filename}: MISSING")
            missing.append(filename)
            all_verified = False
            continue

        actual = compute_sha256(filepath)
        expected_hash = expected.get(filename)

        if expected_hash is None:
            print(f"? {filename}: No expected checksum (computed: {actual[:16]}...)")
            continue

        if actual == expected_hash:
            if args.verbose:
                print(f"✓ {filename}")
                print(f"    SHA256: {actual}")
            else:
                print(f"✓ {filename}: {actual[:16]}... ✓")
            verified.append(filename)
        else:
            print(f"✗ {filename}: CHECKSUM MISMATCH")
            print(f"    Expected: {expected_hash[:16]}...")
            print(f"    Actual:   {actual[:16]}...")
            mismatched.append(filename)
            all_verified = False

    print()
    print("-" * 60)
    
    if all_verified and len(verified) == len(CRITICAL_FILES):
        print("✓ All checksums verified successfully")
        print()
        print("VERIFICATION PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        if missing:
            print(f"✗ Missing files: {missing}")
        if mismatched:
            print(f"✗ Checksum mismatches: {mismatched}")
        print()
        print("VERIFICATION FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
