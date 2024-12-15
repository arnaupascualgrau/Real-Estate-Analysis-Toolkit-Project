from dataclasses import dataclass
from typing import Dict, List, Any
import re

@dataclass
class Cleaner:
    """Class for cleaning real estate data."""
    data: List[Dict[str, Any]]

    def rename_with_best_practices(self) -> None:
        """Rename the columns with best practices (e.g., snake_case and descriptive names)."""
        column_rename_map = {
            "Id": "house_id",
            "SalePrice": "sale_price",
            "LotArea": "lot_area",
            "YearBuilt": "year_built",
            "BedroomAbvGr": "bedroom_above_ground",
            "OverallQual": "overall_qual",
        }

        for row in self.data:
            for key in list(row.keys()):
                new_key = re.sub(r'(?<!^)(?=[A-Z])', '_', key).lower()
                row[new_key] = row.pop(key)

    def na_to_none(self) -> List[Dict[str, Any]]:
        """Replace 'NA' with None in all values."""
        return [
            {k: (None if v == "NA" or v is None else v) for k, v in row.items()}
            for row in self.data
        ]