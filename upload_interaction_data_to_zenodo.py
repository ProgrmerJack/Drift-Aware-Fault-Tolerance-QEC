#!/usr/bin/env python3
"""
upload_interaction_data_to_zenodo.py - Upload interaction effect data to Zenodo

Uploads the new N=69 and N=48 datasets along with interaction analysis results.
"""

import requests
import json
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent

# Zenodo API configuration - read from environment or use token
ZENODO_API_TOKEN = "v0vwEqX8u9dw6MUFZqAQJSGjwcqA3JImFA5zQbPJx4MIJrhlfQgVp77jJz7p"
ZENODO_API_URL = "https://zenodo.org/api/deposit/depositions"

# New files to upload for interaction discovery
NEW_FILES = [
    # Primary dataset (N=69)
    ("results/ibm_experiments/collected_results_20251222_124949.json", "collected_results_N69.json"),
    # Validation dataset (N=48)
    ("results/ibm_experiments/collected_results_20251222_122049.json", "collected_results_N48.json"),
    # Interaction analysis
    ("results/interaction_effect_analysis.json", "interaction_effect_analysis.json"),
    ("results/mechanistic_model.json", "mechanistic_model.json"),
    # New manuscript
    ("manuscript/main_interaction_discovery.tex", "manuscript_interaction_discovery.tex"),
    ("manuscript/supplementary_information.tex", "supplementary_information.tex"),
    # Extended Data figures
    ("manuscript/figures/ExtendedData_Fig1_SessionLevel.pdf", "ExtendedData_Fig1.pdf"),
    ("manuscript/figures/ExtendedData_Fig2_Robustness.pdf", "ExtendedData_Fig2.pdf"),
    ("manuscript/figures/ExtendedData_Fig3_CrossValidation.pdf", "ExtendedData_Fig3.pdf"),
    ("manuscript/figures/ExtendedData_Fig4_Temporal.pdf", "ExtendedData_Fig4.pdf"),
    ("manuscript/figures/fig1_main_interaction.pdf", "Figure1_MainInteraction.pdf"),
]

def get_deposit_info():
    """Load existing deposit info."""
    info_file = PROJECT_ROOT / "zenodo_deposit_info.json"
    if info_file.exists():
        with open(info_file) as f:
            return json.load(f)
    return None

def upload_files_to_existing_deposit():
    """Upload new files to existing Zenodo deposit."""
    deposit_info = get_deposit_info()
    
    if not deposit_info:
        print("ERROR: No existing deposit info found!")
        return False
    
    deposit_id = deposit_info['deposit_id']
    bucket_url = deposit_info['bucket_url']
    
    print(f"Uploading to deposit: {deposit_id}")
    print(f"Bucket URL: {bucket_url}")
    print("=" * 60)
    
    headers = {"Authorization": f"Bearer {ZENODO_API_TOKEN}"}
    
    uploaded = 0
    failed = 0
    
    for local_path, zenodo_name in NEW_FILES:
        full_path = PROJECT_ROOT / local_path
        
        if not full_path.exists():
            print(f"  ⚠ SKIP: {local_path} (not found)")
            continue
        
        print(f"  Uploading: {zenodo_name}...")
        
        try:
            with open(full_path, 'rb') as f:
                response = requests.put(
                    f"{bucket_url}/{zenodo_name}",
                    data=f,
                    headers=headers
                )
            
            if response.status_code == 200 or response.status_code == 201:
                print(f"    ✓ Success")
                uploaded += 1
            else:
                print(f"    ✗ Failed: {response.status_code}")
                print(f"      {response.text[:200]}")
                failed += 1
        except Exception as e:
            print(f"    ✗ Error: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Uploaded: {uploaded} files")
    print(f"Failed: {failed} files")
    
    # Update deposit info
    deposit_info['files_uploaded'] += uploaded
    deposit_info['last_update'] = datetime.now().isoformat()
    deposit_info['interaction_data_added'] = True
    
    with open(PROJECT_ROOT / "zenodo_deposit_info.json", 'w') as f:
        json.dump(deposit_info, f, indent=2)
    
    return failed == 0

def update_metadata():
    """Update Zenodo metadata with interaction effect description."""
    deposit_info = get_deposit_info()
    if not deposit_info:
        return False
    
    deposit_id = deposit_info['deposit_id']
    headers = {
        "Authorization": f"Bearer {ZENODO_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    updated_metadata = {
        "metadata": {
            "title": "Drift-Aware QEC: Interaction Effect Discovery Data",
            "upload_type": "dataset",
            "description": """Supporting data for "Hardware Noise Level Moderates Drift-Aware Quantum Error Correction: An Interaction Effect Reconciling Simulation and Reality"

**KEY DISCOVERY**: Adaptive QEC performance depends critically on hardware noise level (interaction r=0.71, P<10^-11).

## Dataset Contents

### Primary Dataset (N=69)
- `collected_results_N69.json`: 138 QEC jobs (69 paired experiments)
- Collected on IBM Torino, December 22, 2025
- Baseline LER: 0.1076 ± 0.015

### Validation Dataset (N=48)  
- `collected_results_N48.json`: 96 QEC jobs (48 paired experiments)
- Independent replication under different hardware conditions

### Interaction Analysis
- `interaction_effect_analysis.json`: Statistical analysis of the interaction effect
- `mechanistic_model.json`: Overhead model parameters (slope=857.8, intercept=-96.0, R²=0.50)

### Manuscript & Figures
- `manuscript_interaction_discovery.tex`: Main article (Nature Communications format)
- `supplementary_information.tex`: SI with detailed methods
- Extended Data Figures 1-4 (PDF)
- Figure 1 main interaction (PDF)

## Key Statistics
- Interaction correlation: r = 0.711, P < 10^-11
- Low-noise stratum effect: -14.3% (P < 0.0001)
- High-noise stratum effect: +8.3% (P = 0.0001)
- Crossover threshold: LER = 0.112
- Mechanistic model R² = 0.50

## Reproducibility
All analysis code available at https://github.com/ProgrmerJack/Drift-Aware-Fault-Tolerance-QEC""",
            "creators": [
                {"name": "Ashuraliyev, Abduxoliq", "affiliation": "Independent Researcher"}
            ],
            "keywords": [
                "quantum error correction",
                "drift-aware QEC", 
                "interaction effect",
                "IBM quantum",
                "hardware validation"
            ],
            "license": "MIT",
            "related_identifiers": [
                {
                    "identifier": "10.5281/zenodo.14865006",
                    "relation": "isVersionOf",
                    "scheme": "doi"
                }
            ]
        }
    }
    
    response = requests.put(
        f"{ZENODO_API_URL}/{deposit_id}",
        json=updated_metadata,
        headers=headers
    )
    
    if response.status_code == 200:
        print("✓ Metadata updated successfully")
        return True
    else:
        print(f"✗ Metadata update failed: {response.status_code}")
        print(response.text[:500])
        return False

if __name__ == "__main__":
    print("UPLOADING INTERACTION EFFECT DATA TO ZENODO")
    print("=" * 60)
    
    success = upload_files_to_existing_deposit()
    
    if success:
        print("\nUpdating metadata...")
        update_metadata()
        
        print("\n" + "=" * 60)
        print("UPLOAD COMPLETE")
        print("DOI: 10.5281/zenodo.17881116")
        print("NOTE: Publish the deposit to make files publicly accessible")
    else:
        print("\nUpload completed with some failures")
