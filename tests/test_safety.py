from lng_geoenv.safety import safety_override

def make_state(storage, capacity, demand, ships=None, blocked=None):
    return {
        "time_step": 0,
        "storage": {"level": storage, "capacity": capacity},
        "demand_forecast": [demand],
        "ships": ships or [],
        "blocked_routes": blocked or [],
        "budget": 100
    }


def test_no_shortage_passthrough():
    state = make_state(storage=100, capacity=200, demand=50)

    action = {"type": "wait", "parameters": {}}

    result = safety_override(state, action)

    assert result == action


def test_wait_blocked_on_shortage():
    state = make_state(storage=10, capacity=200, demand=100)

    action = {"type": "wait", "parameters": {}}

    result = safety_override(state, action)

    assert result["type"] != "wait"


def test_emergency_triggers_supply_boost():
    state = make_state(storage=5, capacity=200, demand=150)

    action = {"type": "wait", "parameters": {}}

    result = safety_override(state, action)

    assert result["type"] in ["hedge", "reroute", "release"]


def test_reroute_on_blocked_route():
    ships = [
        {"id": 1, "route": "Suez", "eta": 2, "capacity": 100}
    ]

    state = make_state(
        storage=5,
        capacity=200,
        demand=150,
        ships=ships,
        blocked=["Suez"]
    )

    action = {"type": "wait", "parameters": {}}

    result = safety_override(state, action)

    assert result["type"] == "reroute"


def test_store_converted_to_release():
    state = make_state(storage=20, capacity=200, demand=100)

    action = {"type": "store", "parameters": {"amount": 20}}

    result = safety_override(state, action)

    assert result["type"] in ["release", "hedge", "reroute"]