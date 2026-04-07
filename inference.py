import os
from dotenv import load_dotenv

from src.lng_geoenv.env import LNGEnv
from src.lng_geoenv.tasks import get_task_config
from src.lng_geoenv.models import Action
from src.lng_geoenv.agent import LNGAgent
from src.lng_geoenv.evaluator import evaluate_episode

load_dotenv()

MAX_STEPS = 10
TASKS = ["stable", "volatile", "war"]

client = None
try:
    from openai import OpenAI

    API_BASE_URL = os.getenv("API_BASE_URL")
    HF_TOKEN = os.getenv("HF_TOKEN")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

    if API_BASE_URL and HF_TOKEN:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
        )
        print("✅ OpenAI client initialized with API credentials")
    else:
        print(
            "⚠️  No API credentials. Running with baseline policy (expected scores slightly lower)."
        )
except Exception as e:
    print(f"⚠️  Could not initialize OpenAI client: {e}. Running with baseline policy.")
    client = None

if client is None:
    MODEL_NAME = "baseline"


def run_task(task_name):
    """Run inference on a single task configuration."""
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
    agent = LNGAgent(client=client, model_name=MODEL_NAME)
    state = env.reset(seed=42)
    history = []
    step = 0
    done = False

    while not done and step < MAX_STEPS:
        state_dict = state.model_dump()
        action_dict = agent.act(state_dict)
        action = Action(
            action_type=action_dict["type"],
            amount=action_dict["parameters"].get("amount", 0.0),
            ship_id=action_dict["parameters"].get("ship_id"),
            new_route=action_dict["parameters"].get("new_route"),
        )
        state, reward, done, info = env.step(action)
        history.append({"reward": reward.value, "metrics": info.get("metrics", {})})

        print(
            f"  [Step {step + 1:2d}] {action_dict['type']:8s} → reward: {reward.value:7.3f}"
        )

        step += 1

    result = evaluate_episode(history)
    return result


def main():
    """Main entry point: run all tasks and report scores."""

    print("\n" + "=" * 80)
    print("🚀 LNG-GeoEnv Inference")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Steps per episode: {MAX_STEPS}")
    print(f"Tasks: {', '.join(TASKS)}")
    print(f"Seed: 42 (deterministic)")
    print("=" * 80)
    print("\n📋 Expected Baseline Scores (with LLM agent, 10 steps):")
    print("   stable   : 0.765")
    print("   volatile : 0.760")
    print("   war      : 0.615")
    print("   average  : 0.713")
    print("=" * 80)

    all_scores = []

    # Run each task
    for task_name in TASKS:
        print(f"\n📋 Task: {task_name.upper()}")
        print("-" * 80)

        try:
            result = run_task(task_name)
            score = result["final_score"]
            all_scores.append(score)

            print(f"\n✅ Task complete")
            print(f"   Score: {score:.3f}")
            print(f"   Total reward: {result['total_reward']:.2f}")
            print(f"   Steps: {result['steps']}")
            print(f"   Breakdown: {result['breakdown']}")

        except Exception as e:
            print(f"❌ Task failed with error: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("📊 Final Summary")
    print("=" * 80)

    for task_name, score in zip(TASKS, all_scores):
        print(f"  {task_name:10s} : {score:.3f}")

    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        print(f"  {'Average':10s} : {avg_score:.3f}")

    print("=" * 80)
    print("✅ Inference complete\n")


if __name__ == "__main__":
    main()
