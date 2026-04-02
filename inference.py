from src.lng_geoenv.runner import run_task_comparison
import json


def run_all_comparisons():
    tasks = ["stable", "volatile", "war"]
    all_results = []

    for task in tasks:
        print(f"\nRunning comparison for: {task.upper()}")
        print("-" * 80)
        result = run_task_comparison(task, seed=42)
        all_results.append(result)

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

    return all_results


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("LNG-GEOENV: LLM AGENT vs BASELINE POLICY COMPARISON")
    print("=" * 80 + "\n")
    output = run_all_comparisons()
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80 + "\n")
    print(json.dumps(output, indent=2, default=str))
