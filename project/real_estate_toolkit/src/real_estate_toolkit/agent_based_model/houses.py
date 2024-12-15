from enum import Enum
from dataclasses import dataclass
from typing import Optional


class QualityScore(Enum):
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    FAIR = 2
    POOR = 1


@dataclass
class House:
    id: int
    price: float
    area: float
    bedrooms: int
    year_built: int
    data: Optional[dict] = None
    quality_score: Optional[QualityScore] = None
    available: bool = True

    """Calculate and return the price per square foot."""
    def calculate_price_per_square_foot(self) -> float:
        if self.area == 0:
            raise ValueError("Area cannot be zero.")
        return round(self.price / self.area, 2)

    """Determine if house is considered new construction (< 5 years old)."""
    def is_new_construction(self, current_year: int = 2024) -> bool:
        return (current_year - self.year_built) < 5

    """Generate a quality score based on house attributes."""
    def get_quality_score(self) -> QualityScore:
        if self.quality_score:
            return
        score = (self.area / 1000) + (self.bedrooms / 2) - ((2024 - self.year_built) / 10)
        if score >= 4.5:
            self.quality_score = QualityScore.EXCELLENT
        elif score >= 3.5:
            self.quality_score = QualityScore.GOOD
        elif score >= 2.5:
            self.quality_score = QualityScore.AVERAGE
        elif score >= 1.5:
            self.quality_score = QualityScore.FAIR
        else:
            self.quality_score = QualityScore.POOR

    """Mark house as sold."""
    def sell_house(self) -> None:
        self.available = False
