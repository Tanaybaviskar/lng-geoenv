from src.lng_geoenv.runner import run_task
import json


def run_all_tasks():
    tasks = ["stable", "volatile", "war"]

    results = []
    for task in tasks:
        result = run_task(task, seed=42)  # ensure reproducibility
        results.append(result)

from src.lng_geoenv.env import LNGEnv
from src.lng_geoenv.agent import choose_action
from src.lng_geoenv.reward import RewardEngine

# Example config for RewardEngine (adjust as needed)
reward_config = {
    "w_cost": 1.0,
    "w_shortage": 10.0,
    "w_delay": 1.0,
    "w_risk": 1.0,
    "alpha": 2.0,
    "beta": 1.0,
    "gamma": 1.0
}


if __name__ == "__main__":
    output = run_all_tasks()
    print(json.dumps(output, indent=2))
