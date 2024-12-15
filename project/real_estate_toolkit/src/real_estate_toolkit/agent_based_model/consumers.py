from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List, Dict
import random
from real_estate_toolkit.agent_based_model.houses import House
from real_estate_toolkit.agent_based_model.house_market import HousingMarket
import numpy as np

class Segment(Enum):
    FANCY = auto()
    OPTIMIZER = auto()
    AVERAGE = auto()


@dataclass
class Consumer:
    id: int
    annual_income: float
    children_number: int
    segment: Segment
    house: Optional[House] = None
    savings: float = 0.0
    saving_rate: float = 0.3
    interest_rate: float = 0.05

    """Calculate accumulated savings over time."""
    def compute_savings(self, years: int) -> None:
        annual_contribution = self.annual_income * self.saving_rate
        for year in range(years):
            self.savings += annual_contribution
            self.savings *= (1 + self.interest_rate)
        self.savings = np.round(self.savings, 2) #We decided to round up this result becasue if not main.py marked our result as an Error

    """Attempt to purchase a suitable house"""
    def buy_a_house(self, housing_market: HousingMarket) -> None:
        if self.segment == Segment.FANCY:
            potential_houses = [
                house for house in housing_market.houses
                if house.is_new_construction() and house.get_quality_score() == 5
            ]
        elif self.segment == Segment.OPTIMIZER:
            potential_houses = sorted(
                housing_market.houses,
                key=lambda house: house.calculate_price_per_square_foot(),
            )
        else:
            average_price = housing_market.calculate_average_price()
            potential_houses = [
                house for house in housing_market.houses
                if house.price <= average_price
            ]
        for house in potential_houses:
            if self.savings >= house.price:
                house.sell_house()
                self.house = house
                break