import sys
from pathlib import Path

# Fix path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

MAX_STEPS = 10
TASKS = ["stable", "volatile", "war"]

# --- Try imports ---
try:
    from src.lng_geoenv.env import LNGEnv
    from src.lng_geoenv.tasks import get_task_config
    from src.lng_geoenv.models import Action
    from src.lng_geoenv.evaluator import evaluate_episode
    IMPORT_OK = True
except Exception:
    IMPORT_OK = False


def run_dummy(task_name):
    # fallback if imports fail
    for step in range(1, MAX_STEPS + 1):
        sys.stdout.write(f"[STEP] step={step} reward=0.0\n")
        sys.stdout.flush()

    sys.stdout.write(f"[END] task={task_name} score=0.0 steps={MAX_STEPS}\n")
    sys.stdout.flush()


def run_task(task_name):
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

        action = Action(action_type="wait")

        state, reward, done, info = env.step(action)

        history.append({
            "reward": reward.value,
            "metrics": info.get("metrics", {})
        })

        sys.stdout.write(f"[STEP] step={step+1} reward={reward.value}\n")
        sys.stdout.flush()

        step += 1

    result = evaluate_episode(history)

    sys.stdout.write(f"[END] task={task_name} score={result['final_score']} steps={result['steps']}\n")
    sys.stdout.flush()


def main():
    for task_name in TASKS:
        sys.stdout.write(f"[START] task={task_name}\n")
        sys.stdout.flush()

        if not IMPORT_OK:
            run_dummy(task_name)
        else:
            try:
                run_task(task_name)
            except Exception:
                run_dummy(task_name)


if __name__ == "__main__":
    main()