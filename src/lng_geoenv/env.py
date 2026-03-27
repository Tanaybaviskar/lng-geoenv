import random
from .world import update_ships, handle_arrivals


class LNGEnv:
    def __init__(self):
        self.state = None
        self.max_steps = 10

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
            self.state["storage"]["level"] += amt

        elif action["type"] == "reroute":
            for ship in self.state["ships"]:
                if ship["id"] == action["parameters"]["ship_id"]:
                    ship["route"] = action["parameters"]["new_route"]
                    ship["eta"] += 2  # delay

        elif action["type"] == "hedge":
            self.state["budget"] -= 10

        # wait then do nothing

    
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

        # 4. increment time
        self.time_step += 1
        self.state["time_step"] = self.time_step

        done = self.time_step >= self.max_steps
        
        return self.state, done
    
    # literally what it says
    def get_state(self):
        return self.state