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

    env = LNGEnv(config=config, task_config=task_config)
    state = env.reset(seed=seed)

    history = []

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


def run_task_with_llm(task_name, max_steps=10, seed=42):
    """Run task using LLM agent."""
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

    agent = GeminiAgent(use_llm=True)
    history = []
    prev_storage = state.storage.level

    for t in range(max_steps):
        time_step = state.time_step
        demand_forecast = state.demand_forecast
        demand = demand_forecast[min(time_step, len(demand_forecast) - 1)]

        # Get LLM action
        raw_action = agent.choose_action(state.model_dump())
        action = Action(
            **{
                "action_type": raw_action.get("type", raw_action.get("action_type")),
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
            anticipation_score -= 0.5

        state, env_reward, env_done, env_info = env.step(action)

        decision_score = 0.0
        storage_level = state.storage.level
        blocked_routes = state.blocked_routes
        ships = state.ships
        action_type = action.action_type

        ship_on_blocked_route = any(ship.route in blocked_routes for ship in ships)

        # Decision Logic
        if action_type == "reroute" and ship_on_blocked_route:
            decision_score += 1.5
        elif action_type == "reroute":
            decision_score -= 0.5

        if action_type == "wait" and expected_shortage:
            decision_score -= 1.5

        if action_type == "release" and not expected_shortage:
            decision_score -= 0.5

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

        prev_storage = state.storage.level

        if env_done:
            break

    evaluation = evaluate_episode(history)
    return evaluation


def run_task_comparison(task_name, max_steps=10, seed=42):
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

    env_baseline = LNGEnv(config=config, task_config=task_config)
    state_baseline = env_baseline.reset(seed=seed)
    history_baseline = []
    prev_storage_baseline = state_baseline.storage.level

    if COMPARISON_DEBUG:
        print(f"\n{'=' * 80}")
        print(f"POLICY COMPARISON: {task_name.upper()}")
        print(f"{'=' * 80}\n")

    baseline_actions = []

    for t in range(max_steps):
        time_step = state_baseline.time_step
        demand_forecast = state_baseline.demand_forecast
        demand = demand_forecast[min(time_step, len(demand_forecast) - 1)]

        raw_action = choose_action(state_baseline.model_dump(), demand)
        action = Action(
            **{
                "action_type": raw_action.get("type"),
                "amount": raw_action.get("parameters", {}).get("amount", 0.0),
                "ship_id": raw_action.get("parameters", {}).get("ship_id"),
                "new_route": raw_action.get("parameters", {}).get("new_route"),
            }
        )
        validate_action(action)

        baseline_actions.append(action.action_type)

        anticipation_score = 0.0
        expected_shortage = prev_storage_baseline < demand * 1.2
        action_type = action.action_type

        if expected_shortage:
            if action_type in ["reroute", "release", "hedge"]:
                anticipation_score += 1.0
            elif action_type == "wait":
                anticipation_score -= 1.0

        if not expected_shortage and action_type in ["release", "hedge"]:
            anticipation_score -= 0.5

        state_baseline, env_reward, env_done, env_info = env_baseline.step(action)

        decision_score = 0.0
        storage_level = state_baseline.storage.level
        blocked_routes = state_baseline.blocked_routes
        ships = state_baseline.ships
        action_type = action.action_type

        ship_on_blocked_route = any(ship.route in blocked_routes for ship in ships)

        if action_type == "reroute" and ship_on_blocked_route:
            decision_score += 1.5
        elif action_type == "reroute":
            decision_score -= 0.5

        if action_type == "wait" and expected_shortage:
            decision_score -= 1.5

        if action_type == "release" and not expected_shortage:
            decision_score -= 0.5

        if action_type == "release" and expected_shortage:
            decision_score += 1.0

        if action_type == "hedge" and state_baseline.price > 120:
            decision_score += 0.5

        history_baseline.append(
            {
                "state": state_baseline,
                "action": action.model_dump(),
                "reward": env_reward.value,
                "metrics": env_info.get("metrics", {}),
                "decision_score": decision_score,
                "anticipation_score": anticipation_score,
            }
        )

        prev_storage_baseline = state_baseline.storage.level

        if env_done:
            break

    baseline_evaluation = evaluate_episode(history_baseline)

    env_llm = LNGEnv(config=config, task_config=task_config)
    state_llm = env_llm.reset(seed=seed)
    agent = GeminiAgent(use_llm=True)
    history_llm = []
    prev_storage_llm = state_llm.storage.level
    llm_actions = []

    for t in range(max_steps):
        time_step = state_llm.time_step
        demand_forecast = state_llm.demand_forecast
        demand = demand_forecast[min(time_step, len(demand_forecast) - 1)]

        raw_action = agent.choose_action(state_llm.model_dump())
        action = Action(
            **{
                "action_type": raw_action.get("type", raw_action.get("action_type")),
                "amount": raw_action.get("parameters", {}).get("amount", 0.0),
                "ship_id": raw_action.get("parameters", {}).get("ship_id"),
                "new_route": raw_action.get("parameters", {}).get("new_route"),
            }
        )
        validate_action(action)

        llm_actions.append(action.action_type)

        anticipation_score = 0.0
        expected_shortage = prev_storage_llm < demand * 1.2
        action_type = action.action_type

        if expected_shortage:
            if action_type in ["reroute", "release", "hedge"]:
                anticipation_score += 1.0
            elif action_type == "wait":
                anticipation_score -= 1.0

        if not expected_shortage and action_type in ["release", "hedge"]:
            anticipation_score -= 0.5

        state_llm, env_reward, env_done, env_info = env_llm.step(action)

        decision_score = 0.0
        storage_level = state_llm.storage.level
        blocked_routes = state_llm.blocked_routes
        ships = state_llm.ships
        action_type = action.action_type

        ship_on_blocked_route = any(ship.route in blocked_routes for ship in ships)

        if action_type == "reroute" and ship_on_blocked_route:
            decision_score += 1.5
        elif action_type == "reroute":
            decision_score -= 0.5

        if action_type == "wait" and expected_shortage:
            decision_score -= 1.5

        if action_type == "release" and not expected_shortage:
            decision_score -= 0.5

        if action_type == "release" and expected_shortage:
            decision_score += 1.0

        if action_type == "hedge" and state_llm.price > 120:
            decision_score += 0.5

        history_llm.append(
            {
                "state": state_llm,
                "action": action.model_dump(),
                "reward": env_reward.value,
                "metrics": env_info.get("metrics", {}),
                "decision_score": decision_score,
                "anticipation_score": anticipation_score,
            }
        )

        prev_storage_llm = state_llm.storage.level

        if env_done:
            break

    llm_evaluation = evaluate_episode(history_llm)

    # --- STEP-BY-STEP COMPARISON ---
    if COMPARISON_DEBUG:
        print(
            f"{'STEP':<6} {'BASELINE':<12} {'LLM':<12} {'BASELINE R':<12} {'LLM R':<12}"
        )
        print("-" * 80)

        for step in range(min(len(baseline_actions), len(llm_actions))):
            baseline_action = baseline_actions[step]
            llm_action = llm_actions[step]
            baseline_reward = (
                history_baseline[step]["reward"]
                if step < len(history_baseline)
                else 0.0
            )
            llm_reward = history_llm[step]["reward"] if step < len(history_llm) else 0.0

            action_match = "✓" if baseline_action == llm_action else "✗"
            print(
                f"{step:<6} {baseline_action:<12} {llm_action:<12} {baseline_reward:<12.4f} {llm_reward:<12.4f} {action_match}"
            )

        print("-" * 80)

    baseline_score = max(0.0, min(1.0, baseline_evaluation["final_score"]))
    llm_score = max(0.0, min(1.0, llm_evaluation["final_score"]))

    baseline_reward_total = baseline_evaluation["total_reward"]
    llm_reward_total = llm_evaluation["total_reward"]

    score_improvement = (
        ((llm_score - baseline_score) / baseline_score * 100)
        if baseline_score > 0
        else 0
    )
    reward_improvement = (
        ((llm_reward_total - baseline_reward_total) / abs(baseline_reward_total) * 100)
        if baseline_reward_total != 0
        else 0
    )

    if COMPARISON_DEBUG:
        print(f"\n{'=' * 80}")
        print(f"COMPARISON RESULTS - {task_name.upper()}")
        print(f"{'=' * 80}\n")

        print(f"{'METRIC':<30} {'BASELINE':<20} {'LLM':<20}")
        print("-" * 80)
        print(f"{'Final Score':<30} {baseline_score:<20.4f} {llm_score:<20.4f}")
        print(
            f"{'Total Reward':<30} {baseline_reward_total:<20.4f} {llm_reward_total:<20.4f}"
        )
        print(
            f"{'Avg Reward/Step':<30} {baseline_evaluation['avg_reward']:<20.4f} {llm_evaluation['avg_reward']:<20.4f}"
        )
        print(
            f"{'Steps Completed':<30} {baseline_evaluation['steps']:<20} {llm_evaluation['steps']:<20}"
        )
        print("-" * 80)

        # Comparison metrics
        if score_improvement >= 0:
            print(f"\n✅ LLM SCORE IMPROVEMENT: +{score_improvement:.1f}%")
        else:
            print(f"\n⚠️  LLM SCORE CHANGE: {score_improvement:.1f}%")

        if reward_improvement >= 0:
            print(f"✅ LLM REWARD IMPROVEMENT: +{reward_improvement:.1f}%")
        else:
            print(f"⚠️  LLM REWARD CHANGE: {reward_improvement:.1f}%")

        # Action distribution
        baseline_summary = {}
        llm_summary = {}

        for action in baseline_actions:
            baseline_summary[action] = baseline_summary.get(action, 0) + 1

        for action in llm_actions:
            llm_summary[action] = llm_summary.get(action, 0) + 1

        print(f"\n{'Baseline actions:':<30} {baseline_summary}")
        print(f"{'LLM actions:':<30} {llm_summary}")

        print(f"\n{'=' * 80}\n")

    return {
        "task": task_name,
        "baseline": {
            "score": baseline_score,
            "total_reward": baseline_reward_total,
            "avg_reward": baseline_evaluation["avg_reward"],
            "steps": baseline_evaluation["steps"],
            "breakdown": baseline_evaluation["breakdown"],
        },
        "llm": {
            "score": llm_score,
            "total_reward": llm_reward_total,
            "avg_reward": llm_evaluation["avg_reward"],
            "steps": llm_evaluation["steps"],
            "breakdown": llm_evaluation["breakdown"],
        },
        "comparison": {
            "score_improvement_pct": score_improvement,
            "reward_improvement_pct": reward_improvement,
            "baseline_actions": baseline_summary,
            "llm_actions": llm_summary,
        },
    }
