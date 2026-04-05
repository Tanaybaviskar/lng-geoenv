from typing import Dict, Any

CRITICAL_THRESHOLD = 0.25
EMERGENCY_THRESHOLD = 0.10


def get_supply(state):
    storage = state["storage"]["level"]

    ships = state.get("ships", [])
    incoming = sum(
        s["capacity"] for s in ships if s.get("eta", 999) <= 1
    )

    return storage + incoming


def get_demand(state):
    t = state["time_step"]
    return state["demand_forecast"][t]


def detect_shortage(state: Dict[str, Any]):
    supply = get_supply(state)
    demand = get_demand(state)

    deficit = demand - supply

    return {
        "deficit": deficit,
        "ratio": supply / max(demand, 1)
    }


def safety_override(state, action):
    info = detect_shortage(state)

    deficit = info["deficit"]
    ratio = info["ratio"]

    # ✅ No shortage
    if deficit <= 0:
        return action

    # 🚨 EMERGENCY
    if ratio < EMERGENCY_THRESHOLD:
        return emergency_action(state)

    # ⚠️ MODERATE SHORTAGE
    if action["type"] == "wait":
        return emergency_action(state)

    if action["type"] == "store":
        return {"type": "release", "parameters": {"amount": 20}}

    return action


def emergency_action(state):
    ships = state.get("ships", [])
    blocked = state.get("blocked_routes", [])

    # 1. Fix routing first (war task critical)
    for ship in ships:
        if ship["route"] in blocked:
            return {
                "type": "reroute",
                "parameters": {
                    "ship_id": ship["id"],
                    "new_route": "Atlantic"
                }
            }

    # 2. Boost supply
    if state["budget"] >= 10:
        return {"type": "hedge", "parameters": {}}

    # 3. Last fallback
    return {"type": "release", "parameters": {"amount": 20}}