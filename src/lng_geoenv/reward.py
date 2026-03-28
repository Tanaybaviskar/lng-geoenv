class RewardEngine:
    """
    Computes multi-objective reward with strong shortage penalty.
    """

    def __init__(self, config):
        self.w_cost = config["w_cost"]
        self.w_shortage = config["w_shortage"]
        self.w_delay = config["w_delay"]
        self.w_risk = config["w_risk"]

        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.gamma = config["gamma"]

        self.max_penalty = config.get("max_penalty", 1e6)

    def shortage_penalty(self, deficit):
        if deficit <= 0:
            return 0.0
        return min(self.alpha * (deficit ** 2), self.max_penalty) # Quadratic Penalty for better learning

    def delay_penalty(self, delay):
        return self.beta * delay

    def risk_penalty(self, risk, cargo_value):
        return self.gamma * risk * cargo_value

    def cost_penalty(self, fuel, storage, hedge):
        return fuel + storage + hedge

    def compute(self, info):
        C = self.cost_penalty(
            info["fuel_cost"],
            info["storage_cost"],
            info["hedge_cost"]
        )

        S = self.shortage_penalty(info["deficit"])
        D = self.delay_penalty(info["delay"])
        R = self.risk_penalty(info["risk"], info["cargo_value"])

        total = (
            self.w_cost * C +
            self.w_shortage * S +
            self.w_delay * D +
            self.w_risk * R
        )

        reward = -total

        # clamp to avoid exploding gradients
        reward = max(-self.max_penalty, min(self.max_penalty, reward))

        return reward, {
            "cost": C,
            "shortage": S,
            "delay": D,
            "risk": R
        }