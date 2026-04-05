import pytest

from lng_geoenv.demand import DemandGenerator
from lng_geoenv.evaluator import evaluate_episode
from lng_geoenv.tasks import get_task_config
from lng_geoenv.world import World, handle_arrivals, update_ships


def test_demand_generator_is_stateful_and_non_negative():
    generator = DemandGenerator(base=120.0, volatility=0.0, shock_prob=0.0, phi=0.0)

    first = generator.step()
    second = generator.step()

    assert first == 120.0
    assert second == 120.0
    assert first >= 0.0
    assert second >= 0.0


def test_update_ships_and_handle_arrivals():
    ships = [
        {
            "id": 1,
            "origin": "Qatar",
            "destination": "Europe",
            "current_location": "Qatar",
            "eta": 1,
            "capacity": 30.0,
            "route": "Suez",
            "status": "moving",
        },
        {
            "id": 2,
            "origin": "USA",
            "destination": "Europe",
            "current_location": "USA",
            "eta": 2,
            "capacity": 20.0,
            "route": "Panama",
            "status": "moving",
        },
    ]
    storage = {"level": 50.0, "capacity": 100.0}

    updated_ships = update_ships(ships, blocked_routes=[])
    assert updated_ships[0]["status"] == "arrived"
    assert updated_ships[0]["eta"] == 0
    assert updated_ships[1]["eta"] == 1

    ships_after_arrival, updated_storage = handle_arrivals(updated_ships, storage)
    assert ships_after_arrival[0]["status"] == "done"
    assert updated_storage["level"] == 80.0


def test_world_helpers_are_bounded(monkeypatch):
    world = World()
    monkeypatch.setattr("numpy.random.normal", lambda mean, std: 0.0)

    assert world.fuel_cost(100.0) == 5.0
    assert world.route_risk("Suez", ["Suez"]) == pytest.approx(0.9)
    assert world.route_risk("Unknown", []) == 0.5


def test_task_configs_and_unknown_task():
    assert get_task_config("stable")["risk_scale"] == 0.2
    assert get_task_config("volatile")["shock_prob"] == 0.15
    assert get_task_config("war")["seasonal_amp"] == 15

    with pytest.raises(ValueError, match="Unknown task"):
        get_task_config("does-not-exist")


def test_evaluate_episode_returns_expected_summary():
    history = [
        {"reward": -1.0, "metrics": {"cost": 10.0, "shortage": 5.0}},
        {"reward": -2.0, "metrics": {"cost": 4.0, "shortage": 1.0}},
    ]

    result = evaluate_episode(history)

    assert result["steps"] == 2
    assert result["total_reward"] == -3.0
    assert result["breakdown"] == {"cost": 14.0, "shortage": 6.0}
    assert 0.0 <= result["final_score"] <= 1.0