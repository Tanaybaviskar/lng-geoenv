from typing import List, TypedDict


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