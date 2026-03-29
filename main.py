import json
import logging
import sys
import os
from pathlib import Path

from src.lng_geoenv.env import LNGEnv
from src.lng_geoenv.agent import GeminiAgent
from src.lng_geoenv.config import Config, load_env_file

# CONFIGURATION

# Load .env file if it exists
load_env_file(".env")
load_env_file(".env.local")

# Validate configuration
config_status = Config.validate()
if not config_status["valid"]:
    print("❌ Configuration errors:")
    for error in config_status["errors"]:
        print(f"  - {error}")
    sys.exit(1)

if config_status["warnings"]:
    print("⚠️ Configuration warnings:")
    for warning in config_status["warnings"]:
        print(f"  - {warning}")

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.get_log_level()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Environment configuration
env_config = {
    "max_steps": 10,
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


# MAIN EXECUTION


def run_with_llm_agent():
    """
    Main run function using Gemini LLM Agent.
    """
    print("\n" + "=" * 80)
    print("🚀 LNG CRISIS MANAGEMENT WITH GEMINI LLM AGENT")
    print("=" * 80)

    print(f"\n📋 Configuration:")
    print(f"  Agent Enabled: {Config.get_agent_enabled()}")
    print(f"  Model: {Config.get_gemini_model()}")
    print(f"  Temperature: {Config.get_agent_temperature()}")
    print(f"  Debug: {Config.get_debug()}")

    # Initialize environment and agent
    env = LNGEnv(env_config)
    agent = GeminiAgent(use_llm=Config.get_agent_enabled())

    state = env.reset()

    print(f"\n🌍 Environment initialized")
    print(
        f"  Initial Storage: {state['storage']['level']:.1f} / {state['storage']['capacity']:.1f} LNG"
    )
    print(f"  Initial Demand: {state['demand']:.1f} LNG")
    print(f"  Initial Budget: ${state['budget']:.2f}")

    # Tracking
    episode_reward = 0.0
    episode_info = {
        "steps": 0,
        "actions": [],
        "rewards": [],
        "total_cost": 0,
        "total_shortage": 0,
        "total_risk": 0,
        "shortages_occurred": 0,
    }

    print(f"\n▶️  Starting simulation...")
    print("=" * 80)

    for step in range(env_config["max_steps"]):
        print(f"\n📍 STEP {step + 1}/{env_config['max_steps']}")
        print("-" * 80)

        # Get action from LLM agent
        action = agent.choose_action(state)

        print(f"\n🤖 Agent Decision:")
        print(f"  Action: {action['action_type']}")
        print(f"  Reasoning: {action.get('reasoning', 'N/A')}")
        print(f"  Confidence: {action.get('confidence', 0):.2%}")

        # Execute action
        state, reward, done, info = env.step(action)

        print(f"\n📊 Step Results:")
        print(f"  Reward: {reward:.4f}")
        print(
            f"  Storage: {state['storage']['level']:.1f} / {state['storage']['capacity']:.1f} LNG"
        )
        print(f"  Demand: {state['demand']:.1f} LNG")
        print(f"  Price: ${state['price']:.2f}")
        print(f"  Budget: ${state['budget']:.2f}")

        # Track metrics
        episode_reward += reward
        episode_info["steps"] += 1
        episode_info["actions"].append(action["action_type"])
        episode_info["rewards"].append(float(reward))

        if info.get("metrics"):
            metrics = info["metrics"]
            episode_info["total_cost"] += metrics.get("cost", 0)
            episode_info["total_shortage"] += metrics.get("shortage", 0)
            episode_info["total_risk"] += metrics.get("risk", 0)

            if metrics.get("shortage", 0) > 0:
                episode_info["shortages_occurred"] += 1

            print(f"\n📈 Detailed Metrics:")
            print(f"  Cost: {metrics.get('cost', 0):.2f}")
            print(f"  Shortage: {metrics.get('shortage', 0):.2f}")
            print(f"  Delay: {metrics.get('delay', 0):.2f}")
            print(f"  Risk: {metrics.get('risk', 0):.2f}")

        if done:
            print(f"\n✅ Episode finished at step {step + 1}")
            break

    # Final report
    print("\n" + "=" * 80)
    print("📋 FINAL REPORT")
    print("=" * 80)

    avg_reward = episode_reward / max(1, episode_info["steps"])
    final_storage = state["storage"]["level"]

    print(f"\n🏆 Overall Performance:")
    print(f"  Total Reward: {episode_reward:.4f}")
    print(f"  Average Reward per Step: {avg_reward:.4f}")
    print(f"  Total Steps: {episode_info['steps']}")
    print(f"  Final Storage: {final_storage:.1f} LNG")
    print(f"  Final Budget: ${state['budget']:.2f}")

    print(f"\n📊 Aggregated Metrics:")
    print(f"  Total Cost: {episode_info['total_cost']:.2f}")
    print(f"  Total Shortage: {episode_info['total_shortage']:.2f}")
    print(f"  Total Risk: {episode_info['total_risk']:.2f}")
    print(f"  Shortage Events: {episode_info['shortages_occurred']}")

    print(f"\n🎯 Actions Taken:")
    action_counts = {}
    for action in episode_info["actions"]:
        action_counts[action] = action_counts.get(action, 0) + 1
    for action_type, count in sorted(action_counts.items()):
        pct = (
            (count / len(episode_info["actions"]) * 100)
            if episode_info["actions"]
            else 0
        )
        print(f"  {action_type}: {count} ({pct:.1f}%)")

    # JSON output for submission
    print("\n" + "=" * 80)
    print("💾 JSON OUTPUT (for submission)")
    print("=" * 80)

    result = {
        "episode_summary": {
            "total_reward": float(episode_reward),
            "average_reward": float(avg_reward),
            "steps_completed": episode_info["steps"],
            "final_storage": float(final_storage),
            "final_budget": float(state["budget"]),
        },
        "metrics": {
            "total_cost": float(episode_info["total_cost"]),
            "total_shortage": float(episode_info["total_shortage"]),
            "total_risk": float(episode_info["total_risk"]),
            "shortage_events": episode_info["shortages_occurred"],
        },
        "actions": {
            "sequence": episode_info["actions"],
            "distribution": action_counts,
        },
        "agent_config": {
            "model": Config.get_gemini_model(),
            "temperature": Config.get_agent_temperature(),
        },
    }

    print(json.dumps(result, indent=2))

    return result


def run_debug():
    """
    Legacy debug run (without LLM).
    """
    print("\n" + "=" * 80)
    print("🔧 DEBUG RUN (Baseline Policy)")
    print("=" * 80)

    env = LNGEnv(env_config)
    state = env.reset()

    print(f"\n=== DEBUG RUN START ===")

    for step in range(10):
        action = {"type": "wait", "parameters": {}}

        state, reward, done, info = env.step(action)

        print(f"\nStep {step}")
        print("Demand:", round(state["demand"], 2))
        print("Storage:", round(state["storage"]["level"], 2))
        print("Reward:", round(reward, 4))
        print("Metrics:", info.get("metrics", {}))

        if done:
            break

    print("\n=== DEBUG RUN END ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LNG Crisis Management with Gemini Agent"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (baseline policy without LLM)",
    )

    args = parser.parse_args()

    if args.debug:
        run_debug()
    else:
        run_with_llm_agent()
