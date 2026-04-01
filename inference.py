ameer = True
if(ameer == False):
    from dotenv import load_dotenv
    load_dotenv()

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
        # 🔥 LOCAL MODE (NO API KEY NEEDED)
        agent = LNGAgent(
            model_name="local",   # not used for local
            api_key=None,
            use_local=True        # 🔥 IMPORTANT
        )

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

                # 🔥 LLM EVERY STEP
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

        print("\n✅ Local LLM run complete.")


    if __name__ == "__main__":
        main()


