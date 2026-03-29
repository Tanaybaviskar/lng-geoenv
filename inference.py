from src.lng_geoenv.runner import run_task
import json
import numpy as np

def run_all_tasks():
    tasks = ["stable", "volatile", "war"]
    results = [run_task(t) for t in tasks]

    avg_score = sum(r["score"] for r in results) / len(results)

    return {
        "environment": "lng-geoenv",
        "tasks": results,
        "average_score": avg_score,
        "execution_status": "success"
    }

if __name__ == "__main__":
    print(json.dumps(run_all_tasks(), indent=2))