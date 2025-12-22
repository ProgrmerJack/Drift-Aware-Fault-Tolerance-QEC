#!/usr/bin/env python3
"""
download_benchmark.py - Download DAQEC Benchmark Data from Zenodo
==================================================================

Downloads the benchmark dataset if not already present locally.
Verifies checksums after download.

Usage:
    python scripts/download_benchmark.py
    python scripts/download_benchmark.py --force  # Re-download even if exists
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKSUM_FILE = PROJECT_ROOT / "data" / "checksums.json"

# Zenodo configuration
ZENODO_DOI = "10.5281/zenodo.XXXXXXX"  # TODO: Update with actual DOI
ZENODO_RECORD_ID = "XXXXXXX"  # TODO: Update with actual record ID
ZENODO_BASE_URL = f"https://zenodo.org/record/{ZENODO_RECORD_ID}/files"

# Files to download
BENCHMARK_FILES = [
    "master.parquet",
    "SourceData.xlsx",
    "calibration_snapshots.parquet",
    "syndrome_data.tar.gz",
]

# Expected checksums (SHA-256, first 16 chars shown in manifest)
EXPECTED_CHECKSUMS = {
    "master.parquet": None,  # Will be populated from checksums.json
    "SourceData.xlsx": None,
    "calibration_snapshots.parquet": None,
    "syndrome_data.tar.gz": None,
}


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_checksums() -> dict:
    """Load expected checksums from checksums.json."""
    if CHECKSUM_FILE.exists():
        with open(CHECKSUM_FILE) as f:
            return json.load(f)
    return {}


def download_file(url: str, dest: Path) -> bool:
    """Download a file with progress reporting."""
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  ✓ Saved to {dest}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def verify_file(filepath: Path, expected_hash: str) -> bool:
    """Verify file checksum."""
    if not filepath.exists():
        return False
    actual_hash = compute_sha256(filepath)
    if expected_hash and actual_hash != expected_hash:
        print(f"  ✗ Checksum mismatch for {filepath.name}")
        print(f"    Expected: {expected_hash[:16]}...")
        print(f"    Actual:   {actual_hash[:16]}...")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Download DAQEC benchmark data")
    parser.add_argument("--force", action="store_true", help="Re-download even if exists")
    args = parser.parse_args()

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load expected checksums
    checksums = load_checksums()

    print("=" * 60)
    print("DAQEC Benchmark Data Download")
    print("=" * 60)
    print(f"Source: Zenodo DOI {ZENODO_DOI}")
    print(f"Destination: {DATA_DIR}")
    print()

    all_success = True
    for filename in BENCHMARK_FILES:
        filepath = DATA_DIR / filename
        expected_hash = checksums.get(filename)

        # Skip if exists and valid (unless --force)
        if filepath.exists() and not args.force:
            if verify_file(filepath, expected_hash):
                print(f"✓ {filename} already present and valid")
                continue
            else:
                print(f"  Existing file invalid, re-downloading...")

        # Download
        url = f"{ZENODO_BASE_URL}/{filename}"
        if not download_file(url, filepath):
            all_success = False
            continue

        # Verify
        if expected_hash:
            if verify_file(filepath, expected_hash):
                print(f"  ✓ Checksum verified")
            else:
                all_success = False

    print()
    if all_success:
        print("=" * 60)
        print("✓ All benchmark files downloaded and verified")
        print("=" * 60)
        print()
        print("Next step: python scripts/reproduce_all_figures.py")
        sys.exit(0)
    else:
        print("=" * 60)
        print("✗ Some downloads failed or checksums don't match")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
