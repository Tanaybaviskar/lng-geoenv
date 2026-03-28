from typing import List, TypedDict
import numpy as np


class RandomPolicy:
    """
    Baseline random agent for testing.
    """

    def act(self, state):
        return np.array([
            np.random.randint(0, 2),
            np.random.uniform(-20, 20),
            np.random.uniform(0, 30)
        ])

class Ship(TypedDict):
    id: int
    origin: str
    destination: str
    current_location: str
    eta: int
    capacity: float
    route: str
    status: str


class Storage(TypedDict):
    level: float
    capacity: float


class State(TypedDict):
    time_step: int
    ships: List[Ship]
    blocked_routes: List[str]
    storage: Storage
    demand_forecast: List[float]
    price: float
    budget: float