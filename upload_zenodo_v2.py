#!/usr/bin/env python3
"""
Upload Multi-Platform Cross-Validation Package to Zenodo (Version 2)

This script creates a new version of the existing Zenodo record (10.5281/zenodo.18045661)
with the multi-platform cross-validation data.

Usage:
    python upload_zenodo_v2.py --token YOUR_ZENODO_TOKEN

Requirements:
    pip install requests
"""

import argparse
import json
import os
import sys
from pathlib import Path
import requests

# Zenodo API endpoints
ZENODO_API = "https://zenodo.org/api"
EXISTING_RECORD_DOI = "10.5281/zenodo.18045661"
EXISTING_RECORD_ID = "18045661"  # Extract from DOI

def get_access_token():
    """Get Zenodo access token from environment or argument."""
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("ERROR: ZENODO_TOKEN environment variable not set")
        print("Set it with: $env:ZENODO_TOKEN = 'your_token_here'")
        sys.exit(1)
    return token

def create_new_version(token, record_id):
    """Create a new version of an existing Zenodo record."""
    url = f"{ZENODO_API}/deposit/depositions/{record_id}/actions/newversion"
    headers = {"Authorization": f"Bearer {token}"}
    
    print(f"Creating new version of record {record_id}...")
    response = requests.post(url, headers=headers)
    
    if response.status_code == 201:
        data = response.json()
        new_version_url = data.get("links", {}).get("latest_draft")
        if new_version_url:
            # Extract new version ID from URL
            new_id = new_version_url.split("/")[-1]
            print(f"✅ New version created: {new_id}")
            return new_id
        else:
            print(f"❌ Could not extract new version URL from response")
            print(json.dumps(data, indent=2))
            return None
    else:
        print(f"❌ Failed to create new version: {response.status_code}")
        print(response.text)
        return None

def upload_file(token, deposition_id, filepath):
    """Upload a file to a Zenodo deposition."""
    url = f"{ZENODO_API}/deposit/depositions/{deposition_id}/files"
    headers = {"Authorization": f"Bearer {token}"}
    
    filename = os.path.basename(filepath)
    print(f"  Uploading {filename}...")
    
    with open(filepath, "rb") as f:
        data = {"name": filename}
        files = {"file": f}
        response = requests.post(url, headers=headers, data=data, files=files)
    
    if response.status_code == 201:
        print(f"    ✅ {filename} uploaded")
        return True
    else:
        print(f"    ❌ Failed to upload {filename}: {response.status_code}")
        return False

def update_metadata(token, deposition_id):
    """Update the metadata for the new version."""
    url = f"{ZENODO_API}/deposit/depositions/{deposition_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    metadata = {
        "metadata": {
            "title": "Multi-Platform Cross-Validation of Drift-Aware Quantum Error Correction",
            "upload_type": "dataset",
            "description": """<p><strong>Version 2.0 - Multi-Platform Cross-Validation Package</strong></p>

<p>This dataset contains <strong>284 hardware executions</strong> across <strong>4 quantum computing platforms</strong> 
and <strong>2 qubit technologies</strong> (superconducting and trapped-ion), providing comprehensive cross-platform 
validation of the drift-aware QEC interaction effect.</p>

<h3>Key Results by Platform</h3>
<ul>
<li><strong>IBM Torino (133 qubits):</strong> N=69 pairs, interaction r=0.71, P&lt;10⁻¹¹</li>
<li><strong>IBM Fez (156 qubits):</strong> 100% LER improvement, 53.8% depth reduction</li>
<li><strong>IQM Emerald (54 qubits):</strong> N=80 runs, p=0.0485 (statistically significant)</li>
<li><strong>IonQ Forte-1 (36 qubits):</strong> d=18 repetition code (largest hardware distance)</li>
<li><strong>Rigetti Ankaa-3 (82 qubits):</strong> 8 conditions at d=5 and d=9</li>
</ul>

<h3>Contents</h3>
<ul>
<li>Multi-platform validation data (JSON format)</li>
<li>All quantum task identifiers (IBM job IDs, Amazon Braket ARNs)</li>
<li>Statistical analysis results</li>
<li>Comprehensive validation report</li>
</ul>

<p>See MULTI_PLATFORM_CROSS_VALIDATION_README.md for complete documentation.</p>""",
            "creators": [
                {"name": "DAQEC Research Team", "affiliation": "Quantum Computing Research"}
            ],
            "keywords": [
                "quantum error correction",
                "QEC",
                "drift-aware",
                "multi-platform validation",
                "IBM Quantum",
                "IQM",
                "IonQ",
                "Rigetti",
                "trapped-ion",
                "superconducting qubits",
                "cross-platform"
            ],
            "version": "2.0",
            "language": "eng",
            "access_right": "open",
            "license": "MIT",
            "related_identifiers": [
                {
                    "identifier": "https://github.com/ProgrmerJack/Drift-Aware-Fault-Tolerance-QEC",
                    "relation": "isSupplementTo",
                    "scheme": "url"
                }
            ],
            "notes": "Version 2.0 adds comprehensive multi-platform cross-validation data from IQM Emerald, IonQ Forte-1, and Rigetti Ankaa-3, in addition to the original IBM Quantum data."
        }
    }
    
    print("Updating metadata...")
    response = requests.put(url, headers=headers, json=metadata)
    
    if response.status_code == 200:
        print("✅ Metadata updated")
        return True
    else:
        print(f"❌ Failed to update metadata: {response.status_code}")
        print(response.text)
        return False

def publish_record(token, deposition_id):
    """Publish the Zenodo record."""
    url = f"{ZENODO_API}/deposit/depositions/{deposition_id}/actions/publish"
    headers = {"Authorization": f"Bearer {token}"}
    
    print("Publishing record...")
    response = requests.post(url, headers=headers)
    
    if response.status_code == 202:
        data = response.json()
        doi = data.get("doi")
        print(f"✅ Record published! DOI: {doi}")
        return doi
    else:
        print(f"❌ Failed to publish: {response.status_code}")
        print(response.text)
        return None

def main():
    parser = argparse.ArgumentParser(description="Upload multi-platform validation to Zenodo v2")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually upload")
    args = parser.parse_args()
    
    # Get token
    token = get_access_token()
    
    # Define files to upload
    zenodo_dir = Path(__file__).parent / "zenodo_v2"
    files_to_upload = [
        zenodo_dir / "MULTI_PLATFORM_CROSS_VALIDATION_README.md",
        zenodo_dir / "COMPREHENSIVE_VALIDATION_REPORT.md",
        zenodo_dir / "data" / "iqm_validation_CANONICAL_RESULT.json",
        zenodo_dir / "data" / "iqm_v4_combined_analysis.json",
        zenodo_dir / "data" / "ionq_interaction_pair_d18_20251228_133825.json",
        zenodo_dir / "data" / "ionq_interaction_pair_d5_20251228_135923.json",
        zenodo_dir / "data" / "rigetti_validation_20251228_143706.json",
        zenodo_dir / "data" / "rigetti_validation_20251228_143728.json",
        zenodo_dir / "data" / "rigetti_validation_20251228_143832.json",
    ]
    
    # Check files exist
    print("\n📁 Files to upload:")
    for f in files_to_upload:
        if f.exists():
            size = f.stat().st_size / 1024
            print(f"  ✅ {f.name} ({size:.1f} KB)")
        else:
            print(f"  ❌ {f.name} NOT FOUND")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No changes made")
        return
    
    # Create new version
    print(f"\n📤 Creating new version of {EXISTING_RECORD_DOI}...")
    new_id = create_new_version(token, EXISTING_RECORD_ID)
    
    if not new_id:
        print("Failed to create new version. Exiting.")
        sys.exit(1)
    
    # Upload files
    print(f"\n📤 Uploading {len(files_to_upload)} files...")
    for filepath in files_to_upload:
        if filepath.exists():
            upload_file(token, new_id, str(filepath))
    
    # Update metadata
    update_metadata(token, new_id)
    
    # Confirm publish
    print("\n⚠️  Ready to publish. This will create a permanent record.")
    confirm = input("Type 'PUBLISH' to confirm: ")
    
    if confirm == "PUBLISH":
        doi = publish_record(token, new_id)
        if doi:
            print(f"\n🎉 Success! New version available at: https://doi.org/{doi}")
    else:
        print("Publication cancelled. Draft saved - you can publish later from Zenodo web interface.")
        print(f"Draft URL: https://zenodo.org/deposit/{new_id}")

if __name__ == "__main__":
    main()
