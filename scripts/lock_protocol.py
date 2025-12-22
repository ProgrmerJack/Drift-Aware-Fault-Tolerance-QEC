#!/usr/bin/env python3
"""
Lock Protocol Script
====================

Computes SHA-256 hashes of protocol files and creates a locked manifest.
Once run, creates protocol_locked.json with hashes for integrity verification.

Usage:
    python scripts/lock_protocol.py [--verify]
    
Flags:
    --verify    Verify existing lock instead of creating new one
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Protocol directory
PROTOCOL_DIR = Path(__file__).parent.parent / "protocol"
MANIFEST_FILE = PROTOCOL_DIR / "protocol_locked.json"

FILES_TO_LOCK = [
    "protocol.yaml",
    "CLAIMS.md",
]


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_commit() -> str | None:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_git_status() -> dict:
    """Check if working tree is clean."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            return {
                "clean": len(result.stdout.strip()) == 0,
                "modified_files": [
                    line.split()[-1] 
                    for line in result.stdout.strip().split("\n") 
                    if line
                ]
            }
    except Exception:
        pass
    return {"clean": False, "modified_files": []}


def create_manifest() -> dict:
    """Create the lock manifest with hashes."""
    manifest = {
        "protocol_version": "1.0",
        "lock_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "git_status": get_git_status(),
        "files": {},
        "combined_hash": None,
    }
    
    # Hash each file
    combined = hashlib.sha256()
    for filename in FILES_TO_LOCK:
        filepath = PROTOCOL_DIR / filename
        if filepath.exists():
            file_hash = compute_sha256(filepath)
            manifest["files"][filename] = {
                "sha256": file_hash,
                "size_bytes": filepath.stat().st_size,
            }
            combined.update(file_hash.encode())
        else:
            print(f"WARNING: {filename} not found!")
            manifest["files"][filename] = {"sha256": None, "error": "File not found"}
    
    # Combined hash for integrity
    manifest["combined_hash"] = combined.hexdigest()
    
    return manifest


def verify_manifest() -> bool:
    """Verify existing lock manifest."""
    if not MANIFEST_FILE.exists():
        print("ERROR: No lock manifest found. Run without --verify to create one.")
        return False
    
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)
    
    print(f"Verifying protocol locked at: {manifest.get('lock_timestamp')}")
    print(f"Git commit at lock time: {manifest.get('git_commit')}")
    print()
    
    all_valid = True
    combined = hashlib.sha256()
    
    for filename, info in manifest.get("files", {}).items():
        filepath = PROTOCOL_DIR / filename
        expected_hash = info.get("sha256")
        
        if not filepath.exists():
            print(f"  ❌ {filename}: FILE MISSING")
            all_valid = False
            continue
        
        actual_hash = compute_sha256(filepath)
        combined.update(actual_hash.encode())
        
        if actual_hash == expected_hash:
            print(f"  ✓ {filename}: OK")
        else:
            print(f"  ❌ {filename}: MODIFIED")
            print(f"      Expected: {expected_hash[:16]}...")
            print(f"      Actual:   {actual_hash[:16]}...")
            all_valid = False
    
    # Verify combined hash
    actual_combined = combined.hexdigest()
    expected_combined = manifest.get("combined_hash")
    
    print()
    if actual_combined == expected_combined:
        print(f"Combined hash: ✓ OK")
    else:
        print(f"Combined hash: ❌ MISMATCH")
        all_valid = False
    
    print()
    if all_valid:
        print("✓ Protocol integrity verified. Files unchanged since lock.")
    else:
        print("❌ Protocol integrity check FAILED. Files have been modified.")
        print("   If changes are intentional, create AMENDMENT_*.md and re-lock.")
    
    return all_valid


def main():
    if "--verify" in sys.argv:
        success = verify_manifest()
        sys.exit(0 if success else 1)
    
    # Create new manifest
    if MANIFEST_FILE.exists():
        print(f"WARNING: Lock manifest already exists at {MANIFEST_FILE}")
        response = input("Overwrite? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            sys.exit(1)
    
    # Check git status
    git_status = get_git_status()
    if not git_status["clean"]:
        print("WARNING: Git working tree is not clean.")
        print(f"Modified files: {git_status['modified_files']}")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != "yes":
            print("Commit changes first, then re-run.")
            sys.exit(1)
    
    # Create manifest
    manifest = create_manifest()
    
    # Write manifest
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print("=" * 60)
    print("PROTOCOL LOCKED")
    print("=" * 60)
    print(f"Timestamp: {manifest['lock_timestamp']}")
    print(f"Git commit: {manifest['git_commit']}")
    print()
    print("File hashes:")
    for filename, info in manifest["files"].items():
        print(f"  {filename}: {info.get('sha256', 'ERROR')[:32]}...")
    print()
    print(f"Combined manifest hash: {manifest['combined_hash']}")
    print()
    print(f"Manifest saved to: {MANIFEST_FILE}")
    print()
    print("NEXT STEPS:")
    print("  1. Commit this manifest: git add protocol/protocol_locked.json")
    print("  2. Create lock tag: git tag -a protocol_v1_locked -m 'Protocol locked'")
    print("  3. Push tag: git push origin protocol_v1_locked")
    print()
    print("To verify integrity later: python scripts/lock_protocol.py --verify")


if __name__ == "__main__":
    main()
