from typing import List, Optional
from statistics import mean
from real_estate_toolkit.agent_based_model.houses import House


class HousingMarket:
    def __init__(self, houses: List[House]):
        self.houses: List[House] = houses

    """Retrieve specific house by ID."""
    def get_house_by_id(self, house_id: int) -> Optional[House]:
        for house in self.houses:
            if house.id == house_id:
                return house
        return None

    """Calculate average house price, optionally filtered by bedrooms."""
    def calculate_average_price(self, bedrooms: Optional[int] = None) -> float:
        filtered_houses = (
            [house for house in self.houses if house.bedrooms == bedrooms]
            if bedrooms is not None
            else self.houses
        )
        if not filtered_houses:
            raise ValueError("No houses found with the given criteria.")
        return mean(house.price for house in filtered_houses)

    """Filter houses based on buyer requirements."""
    def get_houses_that_meet_requirements(self, max_price: int, segment: str) -> Optional[List[House]]:
        filtered_houses = [
            house for house in self.houses
            if house.price <= max_price and house.available
        ]
        return filtered_houses if filtered_houses else None
