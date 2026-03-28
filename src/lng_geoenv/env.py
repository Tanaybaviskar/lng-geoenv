import random
from .world import update_ships, handle_arrivals
from .demand import DemandGenerator
from .reward import RewardEngine
from .grader import RewardNormalizer

class LNGEnv:
    def __init__(self, config):
        self.state = None
        self.max_steps = config["max_steps"]

        self.demand_gen = DemandGenerator()
        self.reward_engine = RewardEngine(config["reward"])
        self.normalizer = RewardNormalizer()

    # resets the env
    def reset(self, seed=42):
        random.seed(seed)

        self.time_step = 0

        self.state = {
            "time_step": 0,
            "ships": [
                {
                    "id": 1,
                    "origin": "Qatar",
                    "destination": "Europe",
                    "current_location": "Sea",
                    "eta": 5,
                    "capacity": 100,
                    "route": "Suez",
                    "status": "moving",
                },
                {
                    "id": 2,
                    "origin": "USA",
                    "destination": "Europe",
                    "current_location": "Sea",
                    "eta": 7,
                    "capacity": 80,
                    "route": "Panama",
                    "status": "moving",
                }
            ],
            "blocked_routes": random.sample(["Suez", "Panama"], k=1),
            "storage": {
                "level": 50.0,
                "capacity": 200.0,
            },
            "demand_forecast": [],  # Aryan fills this
            "price": random.uniform(50, 150),
            "budget": 500.0,
        }
        self.state["demand"] = self.demand_gen.step()

        # tracking metrics
        self.total_cost = 0
        self.total_shortage = 0
        self.total_risk = 0

        return self.state

    # we can choose from either release, store, reroute, hedge or wait
    def apply_action(self, action):
        self.state["storage"]["level"] = max(
            0,
            self.state["storage"]["level"]
        )
        if action["type"] == "release":
            amt = action["parameters"]["amount"]
            amt = min(amt, self.state["storage"]["level"])
            self.state["storage"]["level"] -= amt

        elif action["type"] == "store":
            amt = action["parameters"]["amount"]
            self.state["storage"]["level"] = min(
                self.state["storage"]["capacity"],
                self.state["storage"]["level"] + amt
            )

        elif action["type"] == "reroute":
            for ship in self.state["ships"]:
                if ship["id"] == action["parameters"]["ship_id"]:
                    ship["route"] = action["parameters"]["new_route"]
                    ship["eta"] += 2  # delay

        elif action["type"] == "hedge":
            self.state["budget"] = max(0, self.state["budget"] - 10)

    
    # how the env changes if i take this step
    def step(self, action):
        # 1. apply action
        self.apply_action(action)

        # 2. update ships
        self.state["ships"] = update_ships(
            self.state["ships"],
            self.state["blocked_routes"]
        )

        # 3. handle arrivals
        self.state["ships"], self.state["storage"] = handle_arrivals(
            self.state["ships"], self.state["storage"]
        )

        # 4. demand evolves
        demand = self.demand_gen.step()
        self.state["demand"] = demand

        # 5. compute supply
        supply = self.state["storage"]["level"]

        deficit = max(0, demand - supply)

        # 6. basic cost model
        fuel_cost = sum([max(0, ship["eta"]) for ship in self.state["ships"]])
        storage_cost = 0.02 * self.state["storage"]["level"]
        hedge_cost = 0 if action["type"] != "hedge" else 10

        # 7. risk model
        risk = 0
        for ship in self.state["ships"]:
            if ship["route"] in self.state["blocked_routes"]:
                risk += 1

        # normalize risk
        risk = risk / max(1, len(self.state["ships"]))

        delay = sum([max(0, ship["eta"]) for ship in self.state["ships"]])

        # 8. reward
        reward, components = self.reward_engine.compute({
            "fuel_cost": fuel_cost,
            "storage_cost": storage_cost,
            "hedge_cost": hedge_cost,
            "deficit": deficit,
            "delay": delay,
            "risk": risk,
            "cargo_value": supply
        })

        reward = self.normalizer.normalize(reward)

        # 9. track totals
        self.total_cost += components["cost"]
        self.total_shortage += components["shortage"]
        self.total_risk += components["risk"]

        # 10. time step
        self.time_step += 1
        self.state["time_step"] = self.time_step

        done = self.time_step >= self.max_steps

        info = {
            "metrics": components,
            "totals": {
                "cost": self.total_cost,
                "shortage": self.total_shortage,
                "risk": self.total_risk
            }
        }

        return self.state, reward, done, info
    
    def get_state(self):
        return self.state