import random
import numpy as np
from .world import update_ships, handle_arrivals, World
from .demand import DemandGenerator
from .reward import RewardEngine
from .grader import RewardNormalizer
from .models import Observation, Action, Reward, Ship, Storage


ROUTE_DISTANCE = {"Suez": 100, "Panama": 140, "Atlantic": 160, "Hormuz": 80}


class LNGEnv:
    def __init__(self, config, task_config=None):
        self.state = None
        self.config = config
        self.task_config = task_config or {}
        self.max_steps = config["max_steps"]

        self.demand_gen = DemandGenerator(
            shock_prob=self.task_config.get("shock_prob", 0.1),
            seasonal_amp=self.task_config.get("seasonal_amp", 0.0),
        )
        self.reward_engine = RewardEngine(config["reward"])
        self.normalizer = RewardNormalizer()
        self.world = World()
        self.world.price_volatility = self.task_config.get("price_volatility", 0.1)
        self.risk_scale = self.task_config.get("risk_scale", 0.3)

    def reset(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        self.time_step = 0

        self.state = Observation(
            time_step=0,
            ships=[
                Ship(
                    id=1,
                    origin="Qatar",
                    destination="Europe",
                    current_location="Qatar",
                    eta=5,
                    capacity=100,
                    route="Suez",
                    status="moving",
                ),
                Ship(
                    id=2,
                    origin="USA",
                    destination="Europe",
                    current_location="USA",
                    eta=7,
                    capacity=80,
                    route="Panama",
                    status="moving",
                ),
            ],
            blocked_routes=random.sample(["Suez", "Panama"], k=1),
            storage=Storage(
                level=50.0,
                capacity=200.0,
            ),
            demand_forecast=[],
            price=random.uniform(50, 150),
            budget=500.0,
        )

        demand = self.demand_gen.step()
        self.state.demand_forecast = [demand]

        self.total_cost = 0
        self.total_shortage = 0
        self.total_risk = 0

        return self.state

    def apply_action(self, action):
        if action.action_type == "release":
            amt = action.amount or 0.0
            amt = min(amt, self.state.storage.level)
            self.state.storage.level -= amt

        elif action.action_type == "store":
            amt = action.amount or 0.0
            self.state.storage.level = min(
                self.state.storage.capacity, self.state.storage.level + amt
            )

        elif action.action_type == "reroute":
            for ship in self.state.ships:
                if action.ship_id is not None and ship.id == action.ship_id:
                    ship.route = action.new_route
                    ship.eta += 2

        elif action.action_type == "hedge":
            cost = 10
            if self.state.budget >= cost:
                self.state.budget -= cost
                self.state.storage.level = min(
                    self.state.storage.capacity, self.state.storage.level + 20
                )  # hedge adds supply

    def step(self, action):
        # Convert dict action to Action model if needed (backward compatibility)
        if isinstance(action, dict):
            action = Action(
                **{
                    "action_type": action.get("type", "wait"),
                    "amount": action.get("parameters", {}).get("amount", 0.0),
                    "ship_id": action.get("parameters", {}).get("ship_id"),
                    "new_route": action.get("parameters", {}).get("new_route"),
                }
            )

        self.apply_action(action)

        ships = update_ships(
            [s.model_dump() for s in self.state.ships], self.state.blocked_routes
        )
        self.state.ships = [Ship(**s) for s in ships]

        ships, storage = handle_arrivals(
            [s.model_dump() for s in self.state.ships], self.state.storage.model_dump()
        )
        self.state.ships = [Ship(**s) for s in ships]
        self.state.storage = Storage(**storage)

        # Demand evolves
        demand = self.demand_gen.step()

        # Price reacts to demand (simple coupling)
        self.state.price *= 1 + 0.01 * (demand / 100)

        self.state.demand_forecast.append(demand)

        # Supply = storage + incoming shipments (ETA <= 1)
        incoming = sum(
            ship.capacity
            for ship in self.state.ships
            if ship.eta <= 1 and ship.status == "moving"
        )

        supply = self.state.storage.level + incoming

        deficit = max(0, demand - supply)

        # Cost model
        fuel_cost = 0
        risk = 0

        for ship in self.state.ships:
            distance = ROUTE_DISTANCE.get(ship.route, 120)
            fuel_cost += self.world.fuel_cost(distance)

            risk += self.world.route_risk(ship.route, self.state.blocked_routes)

        risk /= max(1, len(self.state.ships))

        storage_cost = 0.02 * self.state.storage.level
        hedge_cost = 10 if action.action_type == "hedge" else 0

        delay = sum(max(0, ship.eta) for ship in self.state.ships)

        reward, components = self.reward_engine.compute(
            {
                "fuel_cost": fuel_cost,
                "storage_cost": storage_cost,
                "hedge_cost": hedge_cost,
                "deficit": deficit,
                "delay": delay,
                "risk": risk,
                "cargo_value": supply,
            }
        )

        reward = reward / 1000

        self.total_cost += components["cost"]
        self.total_shortage += components["shortage"]
        self.total_risk += components["risk"]

        self.time_step += 1
        self.state.time_step = self.time_step

        done = self.time_step >= self.max_steps

        return (
            self.state,
            Reward(value=reward, breakdown=components),
            done,
            {"metrics": components},
        )

    def get_state(self):
        return self.state
