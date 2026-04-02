from lng_geoenv.env import LNGEnv
from lng_geoenv.models import Action


def test_reset_initializes_expected_state(env_config, task_config):
    env = LNGEnv(config=env_config, task_config=task_config)
    state = env.reset(seed=42)

    assert state.time_step == 0
    assert len(state.ships) == 2
    assert len(state.blocked_routes) == 1
    assert state.storage.level == 50.0
    assert state.storage.capacity == 200.0
    assert 50.0 <= state.price <= 150.0
    assert state.budget == 500.0
    assert len(state.demand_forecast) == 1


def test_apply_action_updates_state(env_config, task_config):
    env = LNGEnv(config=env_config, task_config=task_config)
    env.reset(seed=42)

    env.apply_action(Action(action_type="release", amount=20.0))
    assert env.state.storage.level == 30.0

    env.apply_action(Action(action_type="store", amount=500.0))
    assert env.state.storage.level == env.state.storage.capacity

    env.apply_action(Action(action_type="hedge"))
    assert env.state.budget == 490.0
    assert env.state.storage.level == env.state.storage.capacity


def test_step_advances_environment_with_deterministic_inputs(monkeypatch, env_config, task_config):
    env = LNGEnv(config=env_config, task_config=task_config)
    env.reset(seed=42)

    monkeypatch.setattr(env.demand_gen, "step", lambda: 110.0)
    monkeypatch.setattr(env.world, "fuel_cost", lambda distance: 5.0)
    monkeypatch.setattr(env.world, "route_risk", lambda route, blocked_routes: 0.25)

    state, reward, done, info = env.step({"type": "wait", "parameters": {}})

    assert state.time_step == 1
    assert state.demand_forecast[-1] == 110.0
    assert done is False
    assert set(info["metrics"].keys()) == {"cost", "shortage", "delay", "risk"}
    assert reward.breakdown == info["metrics"]
    assert isinstance(reward.value, float)