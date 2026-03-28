from lng_geoenv.env import LNGEnv

config = {
    "max_steps": 10,
    "reward": {
        "w_cost": 1.0,
        "w_shortage": 5.0,
        "w_delay": 1.0,
        "w_risk": 1.0,
        "alpha": 2.0,
        "beta": 1.0,
        "gamma": 1.0,
    }
}


def test_env_runs():
    env = LNGEnv(config)
    state = env.reset()

    for _ in range(5):
        action = {"type": "wait", "parameters": {}}
        state, reward, done, info = env.step(action)

    assert state is not None


def test_shortage_occurs():
    env = LNGEnv(config)
    state = env.reset()

    shortages = []

    for _ in range(10):
        action = {"type": "release", "parameters": {"amount": 50}}
        state, reward, done, info = env.step(action)

        shortages.append(info["metrics"]["shortage"])

    assert max(shortages) > 0


def test_storage_bounds():
    env = LNGEnv(config)
    state = env.reset()

    for _ in range(10):
        action = {"type": "store", "parameters": {"amount": 500}}
        state, _, _, _ = env.step(action)

        assert state["storage"]["level"] <= state["storage"]["capacity"]