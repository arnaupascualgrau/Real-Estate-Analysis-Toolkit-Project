from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any
import polars as pl

@dataclass
class DataLoader:
    """Class for loading and basic processing of real estate data."""
    data_path: Path

    def load_data_from_csv(self) -> List[Dict[str, Any]]:
        """Load data from CSV file into a list of dictionaries."""
        try:
            data = pl.read_csv(self.data_path, null_values=["NA", ""])
            return data.to_dicts()
        except Exception as e:
            raise ValueError(f"Error loading data from {self.data_path}: {e}")

    def validate_columns(self, required_columns: List[str]) -> bool:
        """Validate that all required columns are present in the dataset."""
        try:
            data = pl.read_csv(self.data_path, null_values=["NA", ""])
            dataset_columns = set(data.columns)
            return all(col in dataset_columns for col in required_columns)
        except Exception as e:
            raise ValueError(f"Error validating columns in {self.data_path}: {e}")
