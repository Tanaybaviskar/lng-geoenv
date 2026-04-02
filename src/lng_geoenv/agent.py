
from google import genai
import requests


class LNGAgent:
    def __init__(self, model_name: str, api_key: str, use_local=False):
        self.model_name = model_name
        self.use_local = use_local

        if not use_local:
            self.client = genai.Client(api_key=api_key)

    # -----------------------------
    # 🔥 LOCAL LLM (OLLAMA)
    # -----------------------------
    def call_local_llm(self, prompt: str) -> str:
        try:
            res = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "phi4-mini",  
                    "prompt": prompt,
                    "stream": False
                }
            )
            return res.json()["response"].strip().lower()
        except:
            return "wait"

    # -----------------------------
    # 🔥 GEMINI LLM
    # -----------------------------
    def call_gemini(self, prompt: str) -> str:
        try:
            res = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return res.text.strip().lower()
        except:
            return "wait"

    # -----------------------------
    # LLM ACTION
    # -----------------------------
    def get_llm_action(self, state: dict) -> dict:
        t = state["time_step"]
        demand = state["demand_forecast"][t]
        storage = state["storage"]["level"]
        capacity = state["storage"]["capacity"]
        budget = state["budget"]

        ships = state.get("ships", [])
        blocked = state.get("blocked_routes", [])

        incoming = sum(s["capacity"] for s in ships if s.get("eta", 999) <= 1)

        prompt = f"""
You are managing LNG supply optimally.

GOAL:
1. Avoid shortage (MOST IMPORTANT)
2. Minimize cost

STATE:
Demand: {demand}
Storage: {storage}/{capacity}
Incoming: {incoming}
Budget: {budget}
Blocked Routes: {blocked}

RULES:
- release reduces storage
- store/hedge increases supply
- DO NOT cause shortage

Choose ONE:
wait / store / hedge / release_20 / release_50 / reroute

ONLY output action.
"""

        if self.use_local:
            text = self.call_local_llm(prompt)
        else:
            text = self.call_gemini(prompt)

        if "store" in text:
            return {"type": "store", "parameters": {"amount": 20}}
        if "hedge" in text:
            return {"type": "hedge", "parameters": {}}
        if "reroute" in text:
            return {"type": "reroute", "parameters": {"ship_id": 1, "new_route": "Atlantic"}}
        if "50" in text:
            return {"type": "release", "parameters": {"amount": 50}}
        if "20" in text:
            return {"type": "release", "parameters": {"amount": 20}}

        return {"type": "wait", "parameters": {}}

    # -----------------------------
    # BASELINE
    # -----------------------------
    def baseline(self, state: dict) -> dict:
        t = state["time_step"]
        demand = state["demand_forecast"][t]
        storage = state["storage"]["level"]
        capacity = state["storage"]["capacity"]
        budget = state["budget"]

        ships = state.get("ships", [])
        blocked = state.get("blocked_routes", [])

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
                    "parameters": {
                        "ship_id": ship["id"],
                        "new_route": "Atlantic"
                    }
                }

        if storage > 0.85 * capacity and deficit <= 0:
            return {"type": "release", "parameters": {"amount": 20}}

        return {"type": "wait", "parameters": {}}

    # -----------------------------
    # SAFETY
    # -----------------------------
    def safe(self, state: dict, action: dict) -> dict:
        t = state["time_step"]
        demand = state["demand_forecast"][t]
        storage = state["storage"]["level"]
        capacity = state["storage"]["capacity"]

        ships = state.get("ships", [])
        incoming = sum(s["capacity"] for s in ships if s.get("eta", 999) <= 1)

        supply = storage + incoming
        deficit = demand - supply

        if deficit > 0 and action["type"] == "release":
            return self.baseline(state)

        if t == 0 and action["type"] == "reroute":
            return self.baseline(state)

        if action["type"] == "release" and storage < 0.3 * capacity:
            return self.baseline(state)

        if action["type"] == "reroute" and len(state.get("blocked_routes", [])) == 0:
            return self.baseline(state)

        return action

    # -----------------------------
    # FINAL
    # -----------------------------
    def act(self, state: dict) -> dict:
        llm_action = self.get_llm_action(state)
        return self.safe(state, llm_action)
