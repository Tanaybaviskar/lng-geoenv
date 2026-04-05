import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from src.lng_geoenv.env import LNGEnv
from src.lng_geoenv.tasks import get_task_config
from src.lng_geoenv.models import Action
from src.lng_geoenv.agent import LNGAgent
from src.lng_geoenv.evaluator import evaluate_episode


API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# If credentials are missing, run fully offline (baseline policy).
client = None
if API_BASE_URL and HF_TOKEN:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

MAX_STEPS = 20
TASKS = ["stable", "volatile", "war"]


def main():
    if client is None:
        print("⚠️ No API credentials found (HF_TOKEN/API_BASE_URL). Running baseline (no LLM).")

    agent = LNGAgent(client=client, model_name=MODEL_NAME)

    for task in TASKS:
        print(f"\n=== Task: {task} ===")

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

        env = LNGEnv(
            config=config,
            task_config=get_task_config(task)
    import os
    import time

    from src.lng_geoenv.env import LNGEnv
    from src.lng_geoenv.tasks import get_task_config
    from src.lng_geoenv.models import Action
    from src.lng_geoenv.agent import LNGAgent
    from src.lng_geoenv.evaluator import evaluate_episode

    API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL_NAME = "gemini-2.5-flash"

    MAX_STEPS = 20
    TASKS = ["stable", "volatile", "war"]

    # 🔥 LIMITED LLM USAGE
    LLM_STEPS = [0, 5, 10]


    def main():
        agent = LNGAgent(MODEL_NAME, API_KEY)

        for task in TASKS:
            print(f"\n=== Task: {task} ===")

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

            env = LNGEnv(config=config, task_config=get_task_config(task))
            state = env.reset(seed=42)

            history = []
            step = 0
            done = False

            while not done and step < MAX_STEPS:
                state_dict = state.model_dump()

                use_llm = step in LLM_STEPS

                action_dict = agent.act(state_dict, use_llm=use_llm)

                # 🔥 RATE LIMIT SAFE
                if use_llm:
                    time.sleep(12)

                action = Action(
                    action_type=action_dict["type"],
                    amount=action_dict["parameters"].get("amount", 0.0),
                    ship_id=action_dict["parameters"].get("ship_id"),
                    new_route=action_dict["parameters"].get("new_route"),
                )

                state, reward, done, info = env.step(action)

                history.append({
                    "reward": reward.value,
                    "metrics": info.get("metrics", {})
                })

                print(f"[Step {step+1}] {action_dict} → {reward.value:.2f}")

                step += 1

            result = evaluate_episode(history)
            print(f"Score: {result['final_score']:.3f}")

        print("\n✅ Run complete.")


    if __name__ == "__main__":
        main()
else:
    from src.lng_geoenv.env import LNGEnv
    from src.lng_geoenv.tasks import get_task_config
    from src.lng_geoenv.models import Action
    from src.lng_geoenv.agent import LNGAgent
    from src.lng_geoenv.evaluator import evaluate_episode

    MAX_STEPS = 20
    TASKS = ["stable", "volatile", "war"]


    def main():
        agent = LNGAgent(
            model_name="local",
            api_key=None,
            use_local=True
        )

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

            history.append({
                "reward": reward.value,
                "metrics": info.get("metrics", {})
            })

            print(f"[Step {step+1}] {action_dict} → {reward.value:.2f}")

            step += 1

        result = evaluate_episode(history)
        print(f"Score: {result['final_score']:.3f}")

    print("\nRun complete.")


if __name__ == "__main__":
    main()
