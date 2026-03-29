from .env import LNGEnv
from .tasks import get_task_config
from .agent import choose_action
from .evaluator import evaluate_episode

# Debug flag: Set to True for verbose logging
DEBUG = False

def validate_action(action):
    valid_types = ["reroute", "store", "release", "hedge", "wait"]
    assert action["type"] in valid_types, f"Invalid action type: {action['type']}"
    assert isinstance(action["parameters"], dict), "Parameters must be dict"
    return True

def run_task(task_name, max_steps=10, seed=42):
    task_config = get_task_config(task_name)

    config = {
        "max_steps": max_steps,
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

    env = LNGEnv(config)
    state = env.reset(seed=seed)

    history = []

    if DEBUG:
        print(f"\n{'=' * 60}")
        print(f"Task: {task_name.upper()}")
        print(f"{'=' * 60}")
        print(f"Initial State:")
        print(
            f"  Storage: {state.get('storage', {}).get('level', 0.0):.1f} / {state.get('storage', {}).get('capacity', 200.0):.1f}"
        )
        print(f"  Price: ${state.get('price', 100.0):.2f}")
        print(f"  Budget: ${state.get('budget', 500.0):.2f}")
        print()
    prev_storage = state.get("storage", {}).get("level", 0.0)

    for t in range(max_steps):
        time_step = state.get("time_step", t)
        demand_forecast = state.get("demand_forecast", [0.0] * max_steps)
        demand = demand_forecast[min(time_step, len(demand_forecast) - 1)]

        action = choose_action(state, demand)
        validate_action(action)

        # --- Anticipation Evaluation ---
        anticipation_score = 0.0
        expected_shortage = (prev_storage < demand * 1.2)
        action_type = action.get("type")

        if expected_shortage:
            if action_type in ["reroute", "release", "hedge"]:
                anticipation_score += 1.0
            elif action_type == "wait":
                anticipation_score -= 1.0

        if not expected_shortage and action_type in ["release", "hedge"]:
            anticipation_score -= 0.5  # unnecessary action
        state, env_reward, env_done, env_info = env.step(action)

        decision_score = 0.0

        storage_level = state.get("storage", {}).get("level", 0.0)
        blocked_routes = state.get("blocked_routes", [])
        ships = state.get("ships", [])
        action_type = action.get("type")

        # Detect if any ship is actually on blocked route
        ship_on_blocked_route = any(
            ship.get("route") in blocked_routes for ship in ships
        )

        # --- Decision Logic ---

        # Reroute logic
        if action_type == "reroute" and ship_on_blocked_route:
            decision_score += 1.5  # correct proactive reroute

        elif action_type == "reroute":
            decision_score -= 0.5  # unnecessary reroute

        # Waiting during expected shortage
        if action_type == "wait" and expected_shortage:
            decision_score -= 1.5

        # Releasing when not needed
        if action_type == "release" and not expected_shortage:
            decision_score -= 0.5

        # Optional: keep good signals
        if action_type == "release" and expected_shortage:
            decision_score += 1.0

        if action_type == "hedge" and state.get("price", 0) > 120:
            decision_score += 0.5

        history.append({
            "state": state,
            "action": action,
            "reward": env_reward,
            "metrics": env_info.get("metrics", {}),
            "decision_score": decision_score,
            "anticipation_score": anticipation_score
        })

        # update for next step
        prev_storage = state.get("storage", {}).get("level", 0.0)

        if env_done:
            break

    evaluation = evaluate_episode(history)

    if DEBUG:
        print(f"{'=' * 60}")
        print(f"FINAL RESULTS - {task_name.upper()}")
        print(f"{'=' * 60}")
        print(f"Steps completed: {evaluation['steps']}")
        print(f"Total reward: {evaluation['total_reward']:.4f}")
        print(f"Average reward: {evaluation['avg_reward']:.4f}")
        print(f"Final score: {evaluation['final_score']:.4f}")
        print(f"{'=' * 60}\n")

    return {
        "task": task_name,
        "score": evaluation["final_score"],
        "risk_adjusted_score": evaluation["risk_adjusted_score"],
        "breakdown": evaluation["breakdown"],
        "explanation": evaluation["explanation"],
        "total_reward": evaluation["total_reward"],
        "avg_reward": evaluation["avg_reward"],
        "steps": evaluation["steps"],
    }