import json
import numpy as np
from src.lng_geoenv.env import LNGEnv
from src.lng_geoenv.tasks import get_task_config

# Debug flag: Set to True for verbose logging
DEBUG = False


def validate_action(action):
    valid_types = ["reroute", "store", "release", "hedge", "wait"]
    assert action["type"] in valid_types, f"Invalid action type: {action['type']}"
    assert isinstance(action["parameters"], dict), "Parameters must be dict"
    return True


def choose_action(state, demand):
    storage_level = state.get("storage", {}).get("level", 0.0)
    storage_capacity = state.get("storage", {}).get("capacity", 200.0)
    price = state.get("price", 100.0)
    budget = state.get("budget", 500.0)
    ships = state.get("ships", [])
    blocked_routes = state.get("blocked_routes", [])

    valid_routes = ["Suez", "Panama", "Atlantic", "Hormuz"]
    available_routes = [r for r in valid_routes if r not in blocked_routes]

    # Shortage handling: prioritize reroutes for blocked ships
    if storage_level < demand:
        shortage_amount = demand - storage_level

        for ship in ships:
            ship_id = ship.get("id")
            current_route = ship.get("route")

            # Only reroute if on blocked route AND route is actually different
            if (
                ship_id is not None
                and current_route in blocked_routes
                and available_routes
            ):
                # Pick best available route (lowest risk preferred)
                new_route = min(
                    available_routes,
                    key=lambda r: ["Suez", "Panama", "Atlantic", "Hormuz"].index(r),
                )

                if new_route != current_route:
                    return {
                        "type": "reroute",
                        "parameters": {
                            "ship_id": ship_id,
                            "new_route": new_route,
                        },
                    }

        # Release storage if available
        if storage_level > 0:
            release_amount = min(shortage_amount * 0.8, storage_level)
            release_amount = max(0.0, release_amount)
            return {"type": "release", "parameters": {"amount": release_amount}}

    # Reroute ships on blocked routes (defensive)
    for ship in ships:
        ship_id = ship.get("id")
        current_route = ship.get("route")

        if (
            ship_id is not None
            and current_route in blocked_routes
            and ship.get("status") == "moving"
            and available_routes
        ):
            # Pick best available route
            new_route = min(
                available_routes,
                key=lambda r: ["Suez", "Panama", "Atlantic", "Hormuz"].index(r),
            )

            if new_route != current_route:
                return {
                    "type": "reroute",
                    "parameters": {"ship_id": ship_id, "new_route": new_route},
                }

    # Hedge when conditions are favorable (with budget enforcement)
    if price > 120 and budget >= 10:
        return {"type": "hedge", "parameters": {}}

    # Release excess storage
    storage_ratio = storage_level / max(storage_capacity, 1.0)
    if storage_ratio > 0.85:
        release_amount = max(0.0, (storage_level - 0.7 * storage_capacity) * 0.5)
        if release_amount > 0:
            return {"type": "release", "parameters": {"amount": release_amount}}

    return {"type": "wait", "parameters": {}}


def evaluate_episode(history):
    if not history:
        return {
            "total_reward": 0.0,
            "avg_reward": 0.0,
            "final_score": 0.0,
            "steps": 0,
        }

    total_reward = sum(h.get("reward", 0.0) for h in history)
    total_cost = sum(h["metrics"].get("cost", 0.0) for h in history)
    total_shortage = sum(h["metrics"].get("shortage", 0.0) for h in history)
    total_risk = sum(h["metrics"].get("risk", 0.0) for h in history)
    episode_count = len(history)
    avg_reward = total_reward / max(episode_count, 1)

    final_score = np.clip(avg_reward, 0.0, 1.0)
    total_penalty = total_cost + total_shortage + total_risk + 1e-8  # avoid div by zero

    cost_score = total_cost / total_penalty
    shortage_score = total_shortage / total_penalty
    risk_score = total_risk / total_penalty
    
    total_decision_score = sum(h.get("decision_score", 0.0) for h in history)
    # Normalize to [0,1]
    decision_quality = 1 / (1 + np.exp(-total_decision_score / max(1, len(history))))
    
    # --- Generate Explanation ---
    explanation_parts = []

    if shortage_score > 0.5:
        explanation_parts.append("High shortages impacted performance.")

    if cost_score > 0.4:
        explanation_parts.append("Operational costs were significant.")

    if risk_score > 0.3:
        explanation_parts.append("Risk exposure contributed notably.")

    if decision_quality > 0.7:
        explanation_parts.append("Agent demonstrated strong decision-making.")
    elif decision_quality < 0.4:
        explanation_parts.append("Agent decisions were often suboptimal.")

    explanation = " ".join(explanation_parts) if explanation_parts else "Balanced performance across factors."
    
    return {
        "total_reward": total_reward,
        "avg_reward": avg_reward,
        "final_score": final_score,
        "steps": episode_count,
        "breakdown": {
            "cost": cost_score,
            "shortage": shortage_score,
            "risk": risk_score,
            "decision_quality": float(decision_quality)
        },
        "explanation": explanation
    }


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

    for t in range(max_steps):
        time_step = state.get("time_step", t)
        demand_forecast = state.get("demand_forecast", [0.0] * max_steps)
        demand = demand_forecast[min(time_step, len(demand_forecast) - 1)]

        action = choose_action(state, demand)

        validate_action(action)

        state, env_reward, env_done, env_info = env.step(action)

        # --- Decision Quality Evaluation ---
        decision_score = 0.0

        storage_level = state.get("storage", {}).get("level", 0.0)
        blocked_routes = state.get("blocked_routes", [])
        action_type = action.get("type")

        # Good: rerouting blocked ships
        if action_type == "reroute":
            decision_score += 1.0

        # Good: releasing when low storage
        if action_type == "release" and storage_level < demand:
            decision_score += 1.0

        # Good: hedging when price high
        if action_type == "hedge" and state.get("price", 0) > 120:
            decision_score += 0.5

        # Bad: waiting during shortage
        if action_type == "wait" and storage_level < demand:
            decision_score -= 1.0

        history.append({
            "state": state,
            "action": action,
            "reward": env_reward,
            "metrics": env_info.get("metrics", {}),
            "decision_score": decision_score
        })

        if DEBUG and ((t + 1) % 5 == 0 or t == max_steps - 1):
            storage_level = state.get("storage", {}).get("level", 0.0)
            print(f"Step {t + 1}:")
            print(f"  Action: {action['type']}")
            print(f"  Storage: {storage_level:.1f}")
            print(f"  Demand: {demand:.1f}")
            print(f"  Reward: {env_reward:.4f}")
            print()

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
        "breakdown": evaluation["breakdown"],
        "explanation": evaluation["explanation"],
        "total_reward": evaluation["total_reward"],
        "avg_reward": evaluation["avg_reward"],
        "steps": evaluation["steps"],
    }


def run_all_tasks(max_steps=10, seed=42):
    tasks = ["stable", "volatile", "war"]
    results = []

    if DEBUG:
        print("\n" + "=" * 60)
        print("LNG-GeoENV EXECUTION PIPELINE")
        print("=" * 60)

    for task in tasks:
        result = run_task(task, max_steps=max_steps, seed=seed)
        results.append(result)

    if DEBUG:
        print("\n" + "=" * 60)
        print("AGGREGATE RESULTS")
        print("=" * 60)

        avg_score = (
            sum(r["score"] for r in results) / len(results) if results else 0.0
        )
        avg_reward = (
            sum(r["total_reward"] for r in results) / len(results) if results else 0.0
        )

        for result in results:
            print(
                f"{result['task'].upper():10} | Score: {result['score']:.4f} | "
                f"Reward: {result['total_reward']:.4f}"
            )

        print(f"\nAverage Score: {np.clip(avg_score, 0.0, 1.0):.4f}")
        print(f"Average Reward: {np.clip(avg_reward, 0.0, 1.0):.4f}")
        print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    results = run_all_tasks(max_steps=10, seed=42)

    avg_score = (
        sum(r["score"] for r in results) / len(results) if results else 0.0
    )

    output = {
        "environment": "lng-geoenv",
        "tasks": results,
        "average_score": float(np.clip(avg_score, 0.0, 1.0)),
        "execution_status": "success",
    }

    print(json.dumps(output, indent=2))
