from lng_geoenv.models import Action, Observation, Reward, Ship, Storage


def test_observation_demand_tracks_forecast():
    observation = Observation(
        time_step=0,
        ships=[],
        blocked_routes=[],
        storage=Storage(level=10.0, capacity=50.0),
        demand_forecast=[12.5, 13.5],
        price=100.0,
        budget=250.0,
    )

    assert observation.demand == 12.5

    later_observation = observation.model_copy(update={"time_step": 1})
    assert later_observation.demand == 13.5

    past_end_observation = observation.model_copy(update={"time_step": 10})
    assert past_end_observation.demand == 13.5


def test_action_and_reward_models_round_trip():
    action = Action(action_type="release", amount=15.0, ship_id=2, new_route=None)
    reward = Reward(value=-3.25, breakdown={"cost": 1.5})

    assert action.action_type == "release"
    assert action.amount == 15.0
    assert reward.value == -3.25
    assert reward.breakdown == {"cost": 1.5}


def test_ship_and_storage_models_accept_expected_fields():
    ship = Ship(
        id=1,
        origin="Qatar",
        destination="Europe",
        current_location="Qatar",
        eta=5,
        capacity=100.0,
        route="Suez",
        status="moving",
    )
    storage = Storage(level=40.0, capacity=200.0)

    assert ship.capacity == 100.0
    assert storage.level == 40.0