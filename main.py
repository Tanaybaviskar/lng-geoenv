from src.lng_geoenv.env import LNGEnv

config = {
    "max_steps": 10,
    "reward": {
        "w_cost": 1.0,
        "w_shortage": 6.0,
        "w_delay": 1.0,
        "w_risk": 3.0,
        "alpha": 2.0,
        "beta": 1.0,
        "gamma": 2.0,
    }
}


def run_debug():
    env = LNGEnv(config)
    state = env.reset()

    print("\n=== DEBUG RUN START ===")

    for step in range(10):
        action = {"type": "wait", "parameters": {}}

        state, reward, done, info = env.step(action)

        print(f"\nStep {step}")
        print("Demand:", round(state["demand"], 2))
        print("Storage:", round(state["storage"]["level"], 2))
        print("Reward:", round(reward, 4))
        print("Metrics:", info["metrics"])

        if done:
            break

    print("\n=== DEBUG RUN END ===")


if __name__ == "__main__":
    run_debug()