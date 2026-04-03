import os
import google.generativeai as genai
import requests


class LNGAgent:
    def __init__(self, model_name: str, api_key: str, use_local=False):
        self.model_name = model_name
        self.use_local = use_local

        if not use_local and api_key:
            genai.configure(api_key=api_key)

    # -----------------------------
    # 🔥 LOCAL LLM (OLLAMA)
    # -----------------------------
    def call_local_llm(self, prompt: str) -> str:
        try:
            res = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "phi4-mini", "prompt": prompt, "stream": False},
            )
            return res.json()["response"].strip().lower()
        except:
            return "wait"

    # -----------------------------
    # 🔥 GEMINI LLM
    # -----------------------------
    def call_gemini(self, prompt: str) -> str:
        try:
            model = genai.GenerativeModel(self.model_name)
            res = model.generate_content(prompt)
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
            return {
                "type": "reroute",
                "parameters": {"ship_id": 1, "new_route": "Atlantic"},
            }
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
                    "parameters": {"ship_id": ship["id"], "new_route": "Atlantic"},
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


class GeminiAgent:
    def __init__(self, use_llm=True):
        self.use_llm = use_llm
        self.api_key = os.getenv("GEMINI_API_KEY")
        if use_llm and self.api_key:
            genai.configure(api_key=self.api_key)

    def choose_action(self, state: dict) -> dict:
        if not self.use_llm:
            return choose_action(state, state.get("demand_forecast", [0])[0])

        try:
            t = state.get("time_step", 0)
            demand = state.get("demand_forecast", [0])[
                min(t, len(state.get("demand_forecast", [0])) - 1)
            ]
            storage = state.get("storage", {}).get("level", 0)
            capacity = state.get("storage", {}).get("capacity", 100)
            budget = state.get("budget", 0)
            ships = state.get("ships", [])
            blocked = state.get("blocked_routes", [])

            incoming = sum(
                s.get("capacity", 0) for s in ships if s.get("eta", 999) <= 1
            )

            prompt = f"""You are managing LNG supply. Choose ONE action:
Demand: {demand}, Storage: {storage}/{capacity}, Budget: ${budget}
Actions: wait, store, hedge, release, reroute
Reply with just the action name."""

            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            text = response.text.strip().lower()

            if "store" in text:
                return {"type": "store", "parameters": {"amount": 20.0}}
            if "hedge" in text:
                return {"type": "hedge", "parameters": {}}
            if "reroute" in text:
                return {
                    "type": "reroute",
                    "parameters": {"ship_id": 1, "new_route": "Atlantic"},
                }
            if "release" in text:
                return {"type": "release", "parameters": {"amount": 20.0}}
            return {"type": "wait", "parameters": {}}
        except:
            return choose_action(state, state.get("demand_forecast", [0])[0])


def choose_action(state: dict, demand: float) -> dict:
    storage = state.get("storage", {}).get("level", 0)
    capacity = state.get("storage", {}).get("capacity", 100)
    budget = state.get("budget", 0)
    ships = state.get("ships", [])
    blocked = state.get("blocked_routes", [])

    incoming = sum(s.get("capacity", 0) for s in ships if s.get("eta", 999) <= 1)
    supply = storage + incoming
    deficit = demand - supply

    if deficit > 0:
        if budget >= 20:
            return {"type": "store", "parameters": {"amount": 20.0}}
        return {"type": "hedge", "parameters": {}}

    for ship in ships:
        if ship.get("route") in blocked:
            return {
                "type": "reroute",
                "parameters": {"ship_id": ship.get("id", 1), "new_route": "Atlantic"},
            }

    if storage > 0.85 * capacity and deficit <= 0:
        return {"type": "release", "parameters": {"amount": 20.0}}

    return {"type": "wait", "parameters": {}}
