from lng_geoenv.reward import RewardEngine

config = {
    "w_cost": 1,
    "w_shortage": 5,
    "w_delay": 1,
    "w_risk": 1,
    "alpha": 2,
    "beta": 1,
    "gamma": 1,
}


def test_shortage_penalty():
    engine = RewardEngine(config)

    assert engine.shortage_penalty(0) == 0
    assert engine.shortage_penalty(10) > engine.shortage_penalty(5)


def test_risk_penalty():
    engine = RewardEngine(config)

    low = engine.risk_penalty(0.2, 100)
    high = engine.risk_penalty(0.8, 100)

    assert high > low


def test_reward_direction():
    engine = RewardEngine(config)

    good = engine.compute({
        "fuel_cost": 10,
        "storage_cost": 5,
        "hedge_cost": 2,
        "deficit": 0,
        "delay": 1,
        "risk": 0.1,
        "cargo_value": 100
    })[0]

    bad = engine.compute({
        "fuel_cost": 50,
        "storage_cost": 20,
        "hedge_cost": 10,
        "deficit": 20,
        "delay": 10,
        "risk": 0.9,
        "cargo_value": 100
    })[0]

    assert good > bad