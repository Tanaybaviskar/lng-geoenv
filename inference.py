from dotenv import load_dotenv

load_dotenv()

import os
import time
import json
from src.lng_geoenv.runner import run_task
from src.lng_geoenv.env import LNGEnv
from src.lng_geoenv.tasks import get_task_config
from src.lng_geoenv.models import Action
from src.lng_geoenv.agent import GeminiAgent
from src.lng_geoenv.evaluator import evaluate_episode


def run_comparison():
    tasks = ["stable", "volatile", "war"]
    all_results = []

    print("\n" + "=" * 80)
    print("LNG-GEOENV: LLM AGENT vs BASELINE POLICY COMPARISON")
    print("=" * 80)

    for task in tasks:
        baseline_result = run_task(task, seed=42, use_llm=False)
        llm_result = run_task(task, seed=42, use_llm=True)

        baseline_score = baseline_result["score"]
        llm_score = llm_result["score"]
        score_improvement = (
            ((llm_score - baseline_score) / baseline_score * 100)
            if baseline_score > 0
            else 0
        )

        all_results.append(
            {
                "task": task,
                "baseline": baseline_result,
                "llm": llm_result,
                "comparison": {
                    "score_improvement_pct": score_improvement,
                },
            }
        )

    print("\n" + "=" * 80)
    print("SUMMARY: LLM vs BASELINE SCORES")
    print("=" * 80)
    print(f"{'Task':<15} {'Baseline':<15} {'LLM':<15} {'Improvement':<20}")
    print("-" * 80)

    for result in all_results:
        task = result["task"].upper()
        baseline_score = result["baseline"]["score"]
        llm_score = result["llm"]["score"]
        improvement = result["comparison"]["score_improvement_pct"]
        symbol = "✅" if improvement >= 0 else "⚠️ "
        print(
            f"{task:<15} {baseline_score:<15.4f} {llm_score:<15.4f} {symbol} {improvement:+.1f}%"
        )

    print("-" * 80)
    avg_improvement = sum(
        r["comparison"]["score_improvement_pct"] for r in all_results
    ) / len(all_results)
    print(f"\n{'AVERAGE LLM IMPROVEMENT:':<30} {avg_improvement:+.1f}%")
    print("=" * 80 + "\n")

    return all_results


def run_with_limited_llm(max_steps=20):
    tasks = ["stable", "volatile", "war"]
    llm_steps = [0, 5, 10]
    agent = GeminiAgent(use_llm=True)

    for task in tasks:
        print(f"\n=== Task: {task} (LLM at steps {llm_steps}) ===")

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

        env = LNGEnv(config=config, task_config=get_task_config(task))
        state = env.reset(seed=42)

        history = []
        step = 0
        done = False

        while not done and step < max_steps:
            state_dict = state.model_dump()
            use_llm = step in llm_steps

            if use_llm:
                action_dict = agent.choose_action(state_dict)
                time.sleep(12)
            else:
                agent_baseline = GeminiAgent(use_llm=False)
                action_dict = agent_baseline.choose_action(state_dict)

            action = Action(
                action_type=action_dict["type"],
                amount=action_dict["parameters"].get("amount", 0.0),
                ship_id=action_dict["parameters"].get("ship_id"),
                new_route=action_dict["parameters"].get("new_route"),
            )

            state, reward, done, info = env.step(action)

            history.append({"reward": reward.value, "metrics": info.get("metrics", {})})

            print(f"[Step {step + 1}] {action_dict['type']} → {reward.value:.2f}")
            step += 1

        result = evaluate_episode(history)
        print(f"Score: {result['final_score']:.3f}")

    print("\n✅ Limited LLM run complete.")


def run_with_full_llm(max_steps=20):
    tasks = ["stable", "volatile", "war"]
    agent = GeminiAgent(use_llm=True)

    for task in tasks:
        print(f"\n=== Task: {task} (LLM every step) ===")

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

        env = LNGEnv(config=config, task_config=get_task_config(task))
        state = env.reset(seed=42)

        history = []
        step = 0
        done = False

        while not done and step < max_steps:
            state_dict = state.model_dump()
            action_dict = agent.choose_action(state_dict)

            action = Action(
                action_type=action_dict["type"],
                amount=action_dict["parameters"].get("amount", 0.0),
                ship_id=action_dict["parameters"].get("ship_id"),
                new_route=action_dict["parameters"].get("new_route"),
            )

            state, reward, done, info = env.step(action)

            history.append({"reward": reward.value, "metrics": info.get("metrics", {})})

            print(f"[Step {step + 1}] {action_dict['type']} → {reward.value:.2f}")
            step += 1

        result = evaluate_episode(history)
        print(f"Score: {result['final_score']:.3f}")

    print("\n✅ Full LLM run complete.")


if __name__ == "__main__":
    run_comparison()
