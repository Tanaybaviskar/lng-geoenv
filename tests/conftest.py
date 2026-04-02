import pytest

@pytest.fixture
def reward_config():
    return {
        "w_cost": 1.0,
        "w_shortage": 5.0,
        "w_delay": 1.0,
        "w_risk": 2.0,
        "alpha": 2.0,
        "beta": 1.0,
        "gamma": 1.0,
    }


@pytest.fixture
def env_config(reward_config):
    return {
        "max_steps": 3,
        "reward": reward_config,
    }


@pytest.fixture
def task_config():
    return {
        "risk_scale": 0.2,
        "price_volatility": 0.1,
        "shock_prob": 0.0,
        "seasonal_amp": 0.0,
    }