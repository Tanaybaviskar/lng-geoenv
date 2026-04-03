from lng_geoenv.agent import LNGAgent
from lng_geoenv.safety import safety_override


class DummyClient:
    def responses(self, *args, **kwargs):
        return None


def test_agent_never_waits_in_shortage():
    agent = LNGAgent(client=None, model_name="test")

    state = {
        "time_step": 0,
        "storage": {"level": 5, "capacity": 200},
        "demand_forecast": [150],
        "ships": [],
        "blocked_routes": [],
        "budget": 100
    }

    action = agent.baseline(state)

    safe_action = safety_override(state, action)

    assert safe_action["type"] != "wait"


def test_agent_handles_extreme_deficit():
    agent = LNGAgent(client=None, model_name="test")

    state = {
        "time_step": 0,
        "storage": {"level": 0, "capacity": 200},
        "demand_forecast": [200],
        "ships": [],
        "blocked_routes": [],
        "budget": 100
    }

    action = agent.baseline(state)
    safe_action = safety_override(state, action)

    assert safe_action["type"] in ["hedge", "release", "reroute"]