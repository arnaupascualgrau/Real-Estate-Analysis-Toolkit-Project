from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Union
import numpy as np

@dataclass
class Descriptor:
    """Class for describing real estate data."""
    data: List[Dict[str, Any]]

    def _validate_columns(self, columns: Union[List[str], str]) -> List[str]:
        """Validate that column names are correct."""
        valid_columns = self.data[0].keys()
        if columns == "all":
            return list(valid_columns)
        elif isinstance(columns, list):
            invalid_columns = [col for col in columns if col not in valid_columns]
            if invalid_columns:
                raise ValueError(f"Invalid columns: {invalid_columns}")
            return columns
        else:
            raise ValueError("Columns should be 'all' or a list of column names.")

    def none_ratio(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """Compute the ratio of None values per column."""
        columns = self._validate_columns(columns)
        total_rows = len(self.data)
        return {
            col: sum(1 for row in self.data if row.get(col) is None) / total_rows
            for col in columns
        }

    def average(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """Compute the average value for numeric variables."""
        columns = self._validate_columns(columns)
        averages = {}
        for col in columns:
            try:
                values = [row[col] for row in self.data if isinstance(row[col], (int, float))]
                averages[col] = sum(values) / len(values) if values else None
            except Exception as e:
                raise ValueError(f"Error computing average for {col}: {e}")
        return averages

    def median(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """Compute the median value for numeric variables."""
        columns = self._validate_columns(columns)
        medians = {}
        for col in columns:
            values = [row[col] for row in self.data if isinstance(row[col], (int, float))]
            medians[col] = np.median(values) if values else None
        return medians

    def percentile(self, columns: Union[List[str], str] = "all", percentile: int = 50) -> Dict[str, float]:
        """Compute the percentile value for numeric variables."""
        columns = self._validate_columns(columns)
        percentiles = {}
        for col in columns:
            values = [row[col] for row in self.data if isinstance(row[col], (int, float))]
            percentiles[col] = np.percentile(values, percentile) if values else None
        return percentiles

    def type_and_mode(self, columns: Union[List[str], str] = "all") -> Dict[str, Union[Tuple[str, Any]]]:
        """Compute the mode for variables."""
        columns = self._validate_columns(columns)
        modes = {}
        for col in columns:
            values = [row[col] for row in self.data if row[col] is not None]
            mode = max(set(values), key=values.count) if values else None
            modes[col] = (type(values[0]).__name__, mode) if values else None
        return modes


# Now we compute the same calculations with Numpy

class DescriptorNumpy:
    """Class for cleaning real estate data using NumPy."""

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def _validate_columns(self, columns: Union[List[str], str]) -> List[str]:
        """Validate that column names are correct."""
        valid_columns = self.data[0].keys()
        if columns == "all":
            return list(valid_columns)
        elif isinstance(columns, list):
            invalid_columns = [col for col in columns if col not in valid_columns]
            if invalid_columns:
                raise ValueError(f"Invalid columns: {invalid_columns}")
            return columns
        else:
            raise ValueError("Columns should be 'all' or a list of column names.")

    def none_ratio(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """Compute the ratio of None values per column."""
        columns = self._validate_columns(columns)
        total_rows = len(self.data)
        data_array = np.array([[row.get(col) for col in columns] for row in self.data], dtype=object)
        return {
            col: np.sum(data_array[:, i] == None) / total_rows  # noqa: E711
            for i, col in enumerate(columns)
        }

    def average(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """Compute the average value for numeric variables."""
        columns = self._validate_columns(columns)
        averages = {}
        for col in columns:
            values = np.array(
                [row[col] for row in self.data if isinstance(row[col], (int, float))],
                dtype=float
            )
            averages[col] = np.mean(values) if values.size > 0 else None
        return averages

    def median(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """Compute the median value for numeric variables."""
        columns = self._validate_columns(columns)
        medians = {}
        for col in columns:
            values = np.array(
                [row[col] for row in self.data if isinstance(row[col], (int, float))],
                dtype=float
            )
            medians[col] = np.median(values) if values.size > 0 else None
        return medians

    def percentile(self, columns: Union[List[str], str] = "all", percentile: int = 50) -> Dict[str, float]:
        """Compute the percentile value for numeric variables."""
        columns = self._validate_columns(columns)
        percentiles = {}
        for col in columns:
            values = np.array(
                [row[col] for row in self.data if isinstance(row[col], (int, float))],
                dtype=float
            )
            percentiles[col] = np.percentile(values, percentile) if values.size > 0 else None
        return percentiles

    def type_and_mode(self, columns: Union[List[str], str] = "all") -> Dict[str, Union[Tuple[str, Any]]]:
        """Compute the mode for variables."""
        columns = self._validate_columns(columns)
        modes = {}
        for col in columns:
            values = [row[col] for row in self.data if row[col] is not None]
            if values:
                unique, counts = np.unique(values, return_counts=True)
                mode = unique[np.argmax(counts)]
                modes[col] = (type(values[0]).__name__, mode)
            else:
                modes[col] = None
        return modes
