import os
import json

from dotenv import load_dotenv

from src.lng_geoenv.env import LNGEnv
from src.lng_geoenv.tasks import get_task_config
from src.lng_geoenv.models import Action
from src.lng_geoenv.evaluator import evaluate_episode

load_dotenv()

MAX_STEPS = 10
TASKS = ["stable", "volatile", "war"]

client = None
API_BASE_URL = os.getenv("API_BASE_URL", "")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")

try:
    from openai import OpenAI

    if API_BASE_URL and HF_TOKEN:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
        )
        print("✅ OpenAI client initialized with API credentials")
    else:
        print(
            "⚠️  No API credentials (API_BASE_URL / HF_TOKEN). "
            "Running with baseline policy."
        )
except Exception as e:
    print(f"⚠️  Could not initialize OpenAI client: {e}. Running with baseline policy.")
    client = None

if client is None:
    MODEL_NAME = "baseline"


# --- Baseline policy (no LLM) ---
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


# --- LLM-based action selection ---
VALID_ACTIONS = {
    "wait": {"type": "wait", "parameters": {}},
    "store": {"type": "store", "parameters": {"amount": 20}},
    "hedge": {"type": "hedge", "parameters": {}},
    "release_20": {"type": "release", "parameters": {"amount": 20}},
    "release_50": {"type": "release", "parameters": {"amount": 50}},
    "release": {"type": "release", "parameters": {"amount": 20}},
    "reroute": {
        "type": "reroute",
        "parameters": {"ship_id": 1, "new_route": "Atlantic"},
    },
}


def llm_select_action(state_dict):
    if client is None:
        return baseline_policy(state_dict)

    t = state_dict["time_step"]
    demand_val = state_dict["demand_forecast"][t] if t < len(state_dict["demand_forecast"]) else 100

    prompt = f"""You must choose ONE action for an LNG supply chain.

Allowed actions: wait, store, hedge, release_20, release_50, reroute

State:
Demand: {demand_val:.1f}
Storage: {state_dict['storage']['level']:.1f}
Blocked routes: {state_dict.get('blocked_routes', [])}

Rules:
- Output ONLY one word from the list
- Do NOT explain

Answer:"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an LNG supply chain manager. Reply with exactly one action word."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        text = (completion.choices[0].message.content or "").strip().lower()

        # Parse response
        for key in VALID_ACTIONS:
            if key in text:
                return VALID_ACTIONS[key]

        # Fallback to baseline
        return baseline_policy(state_dict)

    except Exception:
        return baseline_policy(state_dict)


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
    state = env.reset(seed=42)
    history = []
    step = 0
    done = False

    while not done and step < MAX_STEPS:
        state_dict = state.model_dump()

        # Select action using LLM or baseline
        action_dict = llm_select_action(state_dict)

        action = Action(
            action_type=action_dict["type"],
            amount=action_dict.get("parameters", {}).get("amount", 0.0),
            ship_id=action_dict.get("parameters", {}).get("ship_id"),
            new_route=action_dict.get("parameters", {}).get("new_route"),
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
    print("START")
    print(f"Model: {MODEL_NAME}")
    print(f"Steps per episode: {MAX_STEPS}")
    print(f"Tasks: {', '.join(TASKS)}")

    for task_name in TASKS:
        try:
            result = run_task(task_name)
            score = result["final_score"]
            print(f"STEP: {task_name}")
            print(f"Score: {score:.3f}")
        except Exception as e:
            print(f"STEP: {task_name}")
            print(f"Score: 0.000")

    print("END")


if __name__ == "__main__":
    main()
