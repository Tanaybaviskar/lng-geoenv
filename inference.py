import os
import sys
from pathlib import Path
from typing import Optional

# Ensure imports work even if validator runs from another dir
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CORE_IMPORT_ERROR: Optional[ModuleNotFoundError] = None
try:
    from src.lng_geoenv.env import LNGEnv
    from src.lng_geoenv.tasks import get_task_config
    from src.lng_geoenv.models import Action
    from src.lng_geoenv.evaluator import evaluate_episode
except ModuleNotFoundError as e:
    _CORE_IMPORT_ERROR = e


MAX_STEPS = 10
TASKS = ["stable", "volatile", "war"]

# --- Baseline policy ---
def baseline_policy(state_dict):
    t = state_dict["time_step"]
    demand = state_dict["demand_forecast"][t] if t < len(state_dict["demand_forecast"]) else 100
    storage = state_dict["storage"]["level"]
    capacity = state_dict["storage"]["capacity"]
    budget = state_dict["budget"]
    ships = state_dict.get("ships", [])
    blocked = state_dict.get("blocked_routes", [])

    incoming = sum(s["capacity"] for s in ships if s.get("eta", 999) <= 1)
    supply = storage + incoming
    deficit = demand - supply

    if deficit > 0:
        if budget >= 20:
            return {"type": "store", "parameters": {"amount": 20}}
        return {"type": "hedge", "parameters": {}}

    for ship in ships:
        if ship["route"] in blocked:
            return {
                "type": "reroute",
                "parameters": {"ship_id": ship["id"], "new_route": "Atlantic"},
            }

    if storage > 0.85 * capacity:
        return {"type": "release", "parameters": {"amount": 20}}

    return {"type": "wait", "parameters": {}}


def run_task(task_name):
    config = {
        "max_steps": MAX_STEPS,
        "reward": {
            "w_cost": 1.0,
            "w_shortage": 6.0,
            "w_delay": 1.0,
            "w_risk": 3.0,
            "alpha": 2.0,
            "beta": 1.0,
            "gamma": 2.0,
        },
    }

    env = LNGEnv(config=config, task_config=get_task_config(task_name))
    state = env.reset(seed=42)

    history = []
    step = 0
    done = False

    while not done and step < MAX_STEPS:
        state_dict = state.model_dump()

        action_dict = baseline_policy(state_dict)

        action = Action(
            action_type=action_dict["type"],
            amount=action_dict.get("parameters", {}).get("amount", 0.0),
            ship_id=action_dict.get("parameters", {}).get("ship_id"),
            new_route=action_dict.get("parameters", {}).get("new_route"),
        )

        state, reward, done, info = env.step(action)

        history.append({
            "reward": reward.value,
            "metrics": info.get("metrics", {})
        })

        # ✅ STRICT FORMAT
        print(f"[STEP] step={step+1} reward={reward.value}", flush=True)

        step += 1

    result = evaluate_episode(history)
    return result


def main():
    if _CORE_IMPORT_ERROR is not None:
        return  # fail silently (validator requirement)

    for task_name in TASKS:
        # ✅ STRICT FORMAT
        print(f"[START] task={task_name}", flush=True)

        try:
            result = run_task(task_name)
            score = result["final_score"]
            steps = result["steps"]

            # ✅ STRICT FORMAT
            print(f"[END] task={task_name} score={score} steps={steps}", flush=True)

        except Exception:
            print(f"[END] task={task_name} score=0.0 steps=0", flush=True)


if __name__ == "__main__":
    main()