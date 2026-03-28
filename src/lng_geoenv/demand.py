import numpy as np


class DemandGenerator:
    """
    Generates time-correlated demand with noise, shocks, and optional seasonality.
    """

    def __init__(
        self,
        base=100.0,
        volatility=10.0,
        shock_prob=0.1,
        shock_range=(20, 50),
        phi=0.7,
        seasonal_amp=0.0,
        seasonal_period=24
    ):
        self.base = base
        self.volatility = volatility
        self.shock_prob = shock_prob
        self.shock_range = shock_range
        self.phi = phi  # persistence factor

        self.seasonal_amp = seasonal_amp
        self.seasonal_period = seasonal_period

        self.prev_demand = base
        self.t = 0

    def step(self):
        # Gaussian noise
        noise = np.random.normal(0, self.volatility)

        # Occasional shock (positive or negative)
        shock = 0.0
        if np.random.rand() < self.shock_prob:
            shock = np.random.uniform(*self.shock_range)
            if np.random.rand() < 0.3:
                shock *= -0.5  # allow demand drops

        # Seasonal component (optional)
        seasonal = self.seasonal_amp * np.sin(
            2 * np.pi * self.t / self.seasonal_period
        )

        # AR(1) process
        demand = (
            self.phi * self.prev_demand +
            (1 - self.phi) * self.base +
            noise + shock + seasonal
        )

        demand = max(0.0, demand)

        self.prev_demand = demand
        self.t += 1

        return demand