import numpy as np

def evaluate_episode(history):
    if not history:
        return {
            "total_reward": 0.0,
            "avg_reward": 0.0,
            "final_score": 0.0,
            "steps": 0,
        }

    total_reward = sum(h.get("reward", 0.0) for h in history)
    total_cost = sum(h["metrics"].get("cost", 0.0) for h in history)
    total_shortage = sum(h["metrics"].get("shortage", 0.0) for h in history)
    total_risk = sum(h["metrics"].get("risk", 0.0) for h in history)
    
    
    total_cost = np.log1p(total_cost)
    total_shortage = np.log1p(total_shortage)
    total_risk = np.log1p(total_risk)

    episode_count = len(history)
    avg_reward = total_reward / max(episode_count, 1)
    final_score = np.clip(avg_reward, 0.0, 1.0)

    # --- Proportional Breakdown ---
    total_penalty = total_cost + total_shortage + total_risk + 1e-8

    cost_score = total_cost / total_penalty
    shortage_score = total_shortage / total_penalty
    risk_score = total_risk / total_penalty

    # --- Decision Quality ---
    total_decision_score = sum(h.get("decision_score", 0.0) for h in history)
    decision_quality = 1 / (1 + np.exp(-total_decision_score / max(1, len(history))))

    # --- Anticipation ---
    total_anticipation = sum(h.get("anticipation_score", 0.0) for h in history)
    anticipation = 1 / (1 + np.exp(-total_anticipation / max(1, len(history))))

    # --- Risk Adjusted Score ---
    risk_adjusted_score = final_score * (1 - risk_score)

    # --- Explanation ---
    explanation_parts = []

    if shortage_score > 0.5:
        explanation_parts.append("High shortages impacted performance.")

    if cost_score > 0.4:
        explanation_parts.append("Operational costs were significant.")

    if risk_score > 0.3:
        explanation_parts.append("Risk exposure contributed notably.")

    if decision_quality > 0.7:
        explanation_parts.append("Agent demonstrated strong decision-making.")
    elif decision_quality < 0.4:
        explanation_parts.append("Agent decisions were often suboptimal.")

    if anticipation > 0.7:
        explanation_parts.append("Agent showed strong anticipatory behavior.")
    elif anticipation < 0.4:
        explanation_parts.append("Agent reacted late to demand changes.")

    explanation = " ".join(explanation_parts) if explanation_parts else "Balanced performance across factors."

    return {
        "total_reward": total_reward,
        "avg_reward": avg_reward,
        "final_score": final_score,
        "risk_adjusted_score": float(risk_adjusted_score),
        "steps": episode_count,
        "breakdown": {
            "cost": cost_score,
            "shortage": shortage_score,
            "risk": risk_score,
            "decision_quality": float(decision_quality),
            "anticipation": float(anticipation)
        },
        "explanation": explanation
    }