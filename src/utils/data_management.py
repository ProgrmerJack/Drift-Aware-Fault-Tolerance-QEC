"""
Utilities Module: Data Management
================================

Data loading, saving, and processing utilities.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import json
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages data storage, loading, and organization for the project.
    
    Handles:
    - Calibration snapshots
    - Probe results
    - QEC experiment results
    - Analysis outputs
    """
    
    def __init__(self, base_dir: str = "data"):
        """
        Initialize data manager.
        
        Args:
            base_dir: Base directory for all data
        """
        self.base_dir = Path(base_dir)
        
        # Create directory structure
        self.dirs = {
            "calibration": self.base_dir / "calibration",
            "probes": self.base_dir / "probes",
            "experiments": self.base_dir / "experiments",
            "analysis": self.base_dir / "analysis",
            "raw": self.base_dir / "raw"
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def save_json(self, data: Dict[str, Any], 
                  category: str,
                  filename: str) -> Path:
        """
        Save data to JSON file.
        
        Args:
            data: Data to save
            category: Data category (calibration, probes, experiments, analysis)
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if category not in self.dirs:
            raise ValueError(f"Unknown category: {category}")
            
        filepath = self.dirs[category] / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        logger.info(f"Saved {category} data to {filepath}")
        return filepath
    
    def load_json(self, category: str, 
                  filename: str) -> Dict[str, Any]:
        """
        Load data from JSON file.
        
        Args:
            category: Data category
            filename: Filename to load
            
        Returns:
            Loaded data dictionary
        """
        filepath = self.dirs[category] / filename
        
        with open(filepath) as f:
            data = json.load(f)
            
        return data
    
    def list_files(self, category: str,
                   pattern: str = "*.json") -> List[Path]:
        """
        List files in a category.
        
        Args:
            category: Data category
            pattern: Glob pattern for filtering
            
        Returns:
            List of file paths
        """
        if category not in self.dirs:
            raise ValueError(f"Unknown category: {category}")
            
        return sorted(self.dirs[category].glob(pattern))
    
    def save_dataframe(self, df: pd.DataFrame,
                       category: str,
                       filename: str,
                       format: str = "parquet") -> Path:
        """
        Save DataFrame to file.
        
        Args:
            df: DataFrame to save
            category: Data category
            filename: Output filename (without extension)
            format: Output format (parquet or csv)
            
        Returns:
            Path to saved file
        """
        if category not in self.dirs:
            raise ValueError(f"Unknown category: {category}")
            
        if format == "parquet":
            filepath = self.dirs[category] / f"{filename}.parquet"
            df.to_parquet(filepath)
        else:
            filepath = self.dirs[category] / f"{filename}.csv"
            df.to_csv(filepath, index=True)
            
        logger.info(f"Saved DataFrame to {filepath}")
        return filepath
    
    def load_dataframe(self, category: str,
                       filename: str) -> pd.DataFrame:
        """
        Load DataFrame from file.
        
        Args:
            category: Data category
            filename: Filename to load
            
        Returns:
            Loaded DataFrame
        """
        if category not in self.dirs:
            raise ValueError(f"Unknown category: {category}")
            
        filepath = self.dirs[category] / filename
        
        if filepath.suffix == ".parquet":
            return pd.read_parquet(filepath)
        else:
            return pd.read_csv(filepath, index_col=0)
    
    def create_dataset(self, name: str,
                       metadata: Optional[Dict[str, Any]] = None) -> Path:
        """
        Create a new dataset directory with metadata.
        
        Args:
            name: Dataset name
            metadata: Optional metadata dictionary
            
        Returns:
            Path to dataset directory
        """
        dataset_dir = self.base_dir / "datasets" / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        meta = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "description": metadata.get("description", "") if metadata else "",
            **(metadata or {})
        }
        
        with open(dataset_dir / "metadata.json", 'w') as f:
            json.dump(meta, f, indent=2)
            
        logger.info(f"Created dataset: {dataset_dir}")
        return dataset_dir
    
    def compute_data_hash(self, filepath: Union[str, Path]) -> str:
        """
        Compute hash of a data file for integrity checking.
        
        Args:
            filepath: Path to file
            
        Returns:
            SHA-256 hash string
        """
        sha256 = hashlib.sha256()
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
                
        return sha256.hexdigest()


class ResultsAggregator:
    """
    Aggregates results from multiple experiments for analysis.
    """
    
    def __init__(self, data_manager: Optional[DataManager] = None):
        """
        Initialize results aggregator.
        
        Args:
            data_manager: Optional DataManager instance
        """
        self.data_manager = data_manager or DataManager()
        
    def aggregate_qec_results(self, 
                               experiment_files: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Aggregate QEC experiment results into a single DataFrame.
        
        Args:
            experiment_files: Optional list of specific files to aggregate
            
        Returns:
            Aggregated results DataFrame
        """
        if experiment_files is None:
            experiment_files = self.data_manager.list_files("experiments", "*.json")
        else:
            experiment_files = [Path(f) for f in experiment_files]
            
        records = []
        
        for filepath in experiment_files:
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    
                # Extract relevant fields
                base_record = {
                    "job_id": data.get("job_id"),
                    "backend": data.get("backend_name", data.get("backend")),
                    "timestamp": data.get("timestamp"),
                    "shots": data.get("shots")
                }
                
                # Extract per-circuit results
                for result in data.get("results", []):
                    circuit_record = base_record.copy()
                    circuit_record["circuit_name"] = result.get("circuit_name")
                    circuit_record["total_shots"] = result.get("total_shots")
                    
                    # Parse circuit name for parameters
                    name = result.get("circuit_name", "")
                    if "d" in name and "r" in name:
                        parts = name.split("_")
                        for part in parts:
                            if part.startswith("d"):
                                circuit_record["distance"] = int(part[1:])
                            elif part.startswith("r"):
                                circuit_record["rounds"] = int(part[1:])
                            elif part.startswith("state"):
                                circuit_record["initial_state"] = int(part[5:])
                                
                    records.append(circuit_record)
                    
            except Exception as e:
                logger.warning(f"Error processing {filepath}: {e}")
                continue
                
        return pd.DataFrame(records)
    
    def aggregate_drift_data(self, 
                              backend_name: str,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Aggregate calibration drift data into time series.
        
        Args:
            backend_name: Backend to aggregate data for
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with drift time series
        """
        calibration_files = self.data_manager.list_files("calibration", f"*{backend_name}*.json")
        
        records = []
        
        for filepath in calibration_files:
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    
                timestamp = datetime.fromisoformat(data.get("timestamp", ""))
                
                # Filter by date
                if start_date and timestamp < start_date:
                    continue
                if end_date and timestamp > end_date:
                    continue
                    
                # Extract qubit properties
                for qubit_key, qubit_data in data.get("qubits", {}).items():
                    qubit_idx = int(qubit_key)
                    
                    record = {
                        "timestamp": timestamp,
                        "qubit": qubit_idx
                    }
                    
                    for prop_name in ["T1", "T2", "readout_error"]:
                        if prop_name in qubit_data:
                            record[prop_name] = qubit_data[prop_name].get("value")
                            
                    records.append(record)
                    
            except Exception as e:
                logger.warning(f"Error processing {filepath}: {e}")
                continue
                
        df = pd.DataFrame(records)
        
        # Pivot for time series analysis
        if not df.empty:
            df = df.set_index(["timestamp", "qubit"]).unstack(level="qubit")
            
        return df
    
    def compute_summary_statistics(self, 
                                    results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute summary statistics from aggregated results.
        
        Args:
            results_df: Aggregated results DataFrame
            
        Returns:
            Summary statistics dictionary
        """
        summary = {
            "total_experiments": len(results_df),
            "backends": results_df["backend"].unique().tolist() if "backend" in results_df else [],
            "date_range": {
                "start": results_df["timestamp"].min() if "timestamp" in results_df else None,
                "end": results_df["timestamp"].max() if "timestamp" in results_df else None
            }
        }
        
        # Group by distance if available
        if "distance" in results_df.columns:
            by_distance = results_df.groupby("distance").size().to_dict()
            summary["experiments_by_distance"] = by_distance
            
        return summary


def setup_data_directories(base_dir: str = "data") -> Dict[str, Path]:
    """
    Set up complete data directory structure.
    
    Args:
        base_dir: Base directory for data
        
    Returns:
        Dictionary of directory paths
    """
    dm = DataManager(base_dir)
    
    # Create additional directories
    additional_dirs = [
        "raw/ibm_quantum",
        "processed",
        "figures/publication",
        "figures/exploratory",
        "datasets",
        "models"
    ]
    
    for subdir in additional_dirs:
        (Path(base_dir) / subdir).mkdir(parents=True, exist_ok=True)
        
    return dm.dirs


if __name__ == "__main__":
    print("Data Management Utilities")
    print("\nSetting up data directories...")
    dirs = setup_data_directories()
    print("Created directories:")
    for name, path in dirs.items():
        print(f"  {name}: {path}")
