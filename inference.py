import numpy as np
from src.lng_geoenv.env import LNGEnv
from src.lng_geoenv.tasks import get_task_config


def choose_action(state, demand):
    """
    Baseline agent policy.
    Returns actions in format: {"type": str, "parameters": dict}
    Ensures all ship_ids are valid and parameters are bounded.
    """
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
            if ship_id is not None and ship.get("route") in blocked_routes:
                if available_routes:
                    new_route = available_routes[0]
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
        if (
            ship_id is not None
            and ship.get("route") in blocked_routes
            and ship.get("status") == "moving"
        ):
            if available_routes:
                new_route = available_routes[0]
                return {
                    "type": "reroute",
                    "parameters": {"ship_id": ship_id, "new_route": new_route},
                }

    # Hedge when conditions are favorable
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
    """
    Evaluates episode using rewards already normalized to [0, 1] by env.step().
    Returns final_score in [0.0, 1.0].
    """
    if not history:
        return {
            "total_reward": 0.0,
            "avg_reward": 0.0,
            "final_score": 0.0,
            "steps": 0,
        }

    total_reward = sum(h.get("reward", 0.0) for h in history)
    episode_count = len(history)
    avg_reward = total_reward / max(episode_count, 1)

    # Rewards are already normalized [0, 1], so average is final score
    final_score = np.clip(avg_reward, 0.0, 1.0)

    return {
        "total_reward": total_reward,
        "avg_reward": avg_reward,
        "final_score": final_score,
        "steps": episode_count,
    }


def run_task(task_name, max_steps=10, seed=42):
    """
    Executes a single task (stable, volatile, or war).
    Uses env.step() which provides normalized [0, 1] rewards.
    """
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
        demand = state.get("demand", 0.0)

        action = choose_action(state, demand)

        state, env_reward, env_done, env_info = env.step(action)

        # env_reward is already normalized [0, 1] by RewardNormalizer in env.step()
        history.append({"state": state, "action": action, "reward": env_reward})

        if (t + 1) % 5 == 0 or t == max_steps - 1:
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
        "final_score": evaluation["final_score"],
        "total_reward": evaluation["total_reward"],
        "avg_reward": evaluation["avg_reward"],
        "steps": evaluation["steps"],
    }


def run_all_tasks(max_steps=10, seed=42):
    """
    Executes all three tasks and aggregates results.
    Final scores and rewards are guaranteed in [0, 1].
    """
    tasks = ["stable", "volatile", "war"]
    results = []

    print("\n" + "=" * 60)
    print("LNG-GeoENV EXECUTION PIPELINE")
    print("=" * 60)

    for task in tasks:
        result = run_task(task, max_steps=max_steps, seed=seed)
        results.append(result)

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    avg_score = (
        sum(r["final_score"] for r in results) / len(results) if results else 0.0
    )
    avg_reward = (
        sum(r["total_reward"] for r in results) / len(results) if results else 0.0
    )

    for result in results:
        print(
            f"{result['task'].upper():10} | Score: {result['final_score']:.4f} | "
            f"Reward: {result['total_reward']:.4f}"
        )

    print(f"\nAverage Score: {np.clip(avg_score, 0.0, 1.0):.4f}")
    print(f"Average Reward: {np.clip(avg_reward, 0.0, 1.0):.4f}")
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    results = run_all_tasks(max_steps=10, seed=42)
