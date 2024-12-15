from enum import Enum, auto
from dataclasses import dataclass
from random import gauss, randint, shuffle, random
from typing import List, Dict, Any, Optional
from real_estate_toolkit.agent_based_model.houses import House
from real_estate_toolkit.agent_based_model.house_market import HousingMarket
from real_estate_toolkit.agent_based_model.consumers import Consumer, Segment
import random 
import numpy as np

class CleaningMarketMechanism(Enum):
    INCOME_ORDER_DESCENDANT = auto()
    INCOME_ORDER_ASCENDANT = auto()
    RANDOM = auto()


@dataclass
class AnnualIncomeStatistics:
    minimum: float
    average: float
    standard_deviation: float
    maximum: float


@dataclass
class ChildrenRange:
    minimum: int = 0
    maximum: int = 5


@dataclass
class Simulation:
    housing_market_data: List[Dict[str, Any]]
    consumers_number: int
    years: int
    annual_income: AnnualIncomeStatistics
    children_range: ChildrenRange
    cleaning_market_mechanism: CleaningMarketMechanism
    down_payment_percentage: float
    saving_rate: float
    interest_rate: float
    housing_market: Optional[HousingMarket] = None
    consumers: List[Consumer] = None

    """Initialize market with houses."""
    def create_housing_market(self) -> None:
        houses = [House(id=data['id'], price=data['sale_price'], area=data['gr_liv_area'], bedrooms=data['bedroom_abv_gr'], year_built=data['year_built'] , data=data) for data in self.housing_market_data]
        self.housing_market = HousingMarket(houses)

    """ Generate consumer population."""
    def create_consumers(self) -> None:
        self.consumers = []
        for _ in range(self.consumers_number):
            annual_income = gauss(
                self.annual_income.average, self.annual_income.standard_deviation
            )
            while annual_income < self.annual_income.minimum or annual_income > self.annual_income.maximum:
                annual_income = gauss(
                    self.annual_income.average, self.annual_income.standard_deviation
                )
            children_number = randint(
                self.children_range.minimum, self.children_range.maximum
            )
            segment = random.choice(list(Segment)) # Using random.choice from random library
            self.consumers.append(
                Consumer(
                    id=len(self.consumers),
                    annual_income=annual_income,
                    children_number=children_number,
                    segment=segment,
                )
            )
    """Execute market transactions."""
    def clean_the_market(self) -> None:
        if self.cleaning_market_mechanism == CleaningMarketMechanism.INCOME_ORDER_DESCENDANT:
            self.consumers.sort(key=lambda c: c.annual_income, reverse=True)
        elif self.cleaning_market_mechanism == CleaningMarketMechanism.INCOME_ORDER_ASCENDANT:
            self.consumers.sort(key=lambda c: c.annual_income)
        else:
            shuffle(self.consumers)
        for consumer in self.consumers:
            consumer.buy_a_house(self.housing_market)

    """Calculate savings for all consumers."""
    def compute_consumers_savings(self) -> None:
        """Compute savings for all consumers in the simulation."""
        for consumer in self.consumers:
            consumer.compute_savings(self.years) 

    """Compute the owners population rate after the market is clean."""
    def compute_owners_population_rate(self) -> float:
        """Compute the percentage of consumers who own a house."""
        return sum(1 for consumer in self.consumers if consumer.house is not None) / len(self.consumers)

    """Compute the houses availability rate after the market is clean."""
    def compute_houses_availability_rate(self) -> float:
        """Compute the percentage of available houses in the market."""
        return sum(1 for house in self.housing_market.houses if house.available) / len(self.housing_market.houses)