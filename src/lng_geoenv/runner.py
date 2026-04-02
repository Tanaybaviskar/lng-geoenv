from .env import LNGEnv
from .tasks import get_task_config
from .agent import choose_action, GeminiAgent
from .models import Action
from .evaluator import evaluate_episode

# Debug flag: Set to True for verbose logging
DEBUG = False
COMPARISON_DEBUG = True  # Compare LLM vs Baseline


def validate_action(action: Action):
    valid_types = ["reroute", "store", "release", "hedge", "wait"]
    assert action.action_type in valid_types, (
        f"Invalid action type: {action.action_type}"
    )
    return True


def run_task(task_name, max_steps=10, seed=42, use_llm=False):
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

    env = LNGEnv(config=config, task_config=task_config)
    state = env.reset(seed=seed)

    history = []
    agent = GeminiAgent(use_llm=True) if use_llm else None

    if DEBUG:
        print(f"\n{'=' * 60}")
        print(f"Task: {task_name.upper()}")
        print(f"{'=' * 60}")
        print(f"Initial State:")
        print(f"  Storage: {state.storage.level:.1f} / {state.storage.capacity:.1f}")
        print(f"  Price: ${state.price:.2f}")
        print(f"  Budget: ${state.budget:.2f}")
        print()
    prev_storage = state.storage.level

    for t in range(max_steps):
        time_step = state.time_step
        demand_forecast = state.demand_forecast
        demand = demand_forecast[min(time_step, len(demand_forecast) - 1)]

        if use_llm:
            raw_action = agent.choose_action(state.model_dump())
        else:
            raw_action = choose_action(state.model_dump(), demand)
        action = Action(
            **{
                "action_type": raw_action.get("type"),
                "amount": raw_action.get("parameters", {}).get("amount", 0.0),
                "ship_id": raw_action.get("parameters", {}).get("ship_id"),
                "new_route": raw_action.get("parameters", {}).get("new_route"),
            }
        )
        validate_action(action)

        # --- Anticipation Evaluation ---
        anticipation_score = 0.0
        expected_shortage = prev_storage < demand * 1.2
        action_type = action.action_type

        if expected_shortage:
            if action_type in ["reroute", "release", "hedge"]:
                anticipation_score += 1.0
            elif action_type == "wait":
                anticipation_score -= 1.0

        if not expected_shortage and action_type in ["release", "hedge"]:
            anticipation_score -= 0.5  # unnecessary action
        state, env_reward, env_done, env_info = env.step(action)

        decision_score = 0.0

        storage_level = state.storage.level
        blocked_routes = state.blocked_routes
        ships = state.ships
        action_type = action.action_type

        # Detect if any ship is actually on blocked route
        ship_on_blocked_route = any(ship.route in blocked_routes for ship in ships)

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

        if action_type == "hedge" and state.price > 120:
            decision_score += 0.5

        history.append(
            {
                "state": state,
                "action": action.model_dump(),
                "reward": env_reward.value,
                "metrics": env_info.get("metrics", {}),
                "decision_score": decision_score,
                "anticipation_score": anticipation_score,
            }
        )

        # update for next step
        prev_storage = state.storage.level

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
