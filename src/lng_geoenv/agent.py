
from google import genai

VALID_ACTIONS = ["reroute", "store", "release", "hedge", "wait"]

class LNGAgent:
    def __init__(self, model_name: str, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    # -----------------------------
    # 🔥 FIXED BASELINE POLICY
    # -----------------------------
    def _baseline_action(self, state: dict) -> dict:
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

        # --- KEY CORRECT LOGIC ---

        # 1. If deficit → increase supply (NOT release)
        if deficit > 0:
            if budget >= 20:
                return {"type": "store", "parameters": {"amount": 20}}
            return {"type": "hedge", "parameters": {}}

        # 2. Reroute blocked ships (future supply)
        for ship in ships:
            if ship["route"] in blocked:
                return {
                    "type": "reroute",
                    "parameters": {
                        "ship_id": ship["id"],
                        "new_route": "Atlantic"
                    }
                }

        # 3. Avoid over-storage
        if storage > 0.85 * capacity:
            return {"type": "release", "parameters": {"amount": 20}}

        return {"type": "wait", "parameters": {}}

    # -----------------------------
    # 🔥 FIXED SIMULATION (CRITICAL)
    # -----------------------------
    def _simulate_step(self, state: dict, action: dict):
        next_state = dict(state)

        storage = next_state["storage"]["level"]
        capacity = next_state["storage"]["capacity"]

        # Apply action (same semantics as env)
        if action["type"] == "release":
            amt = action["parameters"].get("amount", 0)
            storage = max(0, storage - amt)

        elif action["type"] == "store":
            amt = action["parameters"].get("amount", 0)
            storage = min(capacity, storage + amt)

        elif action["type"] == "hedge":
            storage += 20

        # Approximate next demand
        t = state["time_step"]
        demand = state["demand_forecast"][t]

        ships = state.get("ships", [])
        incoming = sum(s["capacity"] for s in ships if s.get("eta", 999) <= 1)

        supply = storage + incoming
        deficit = max(0, demand - supply)

        return {
            "storage": storage,
            "deficit": deficit
        }

    # -----------------------------
    # LLM (optional guidance)
    # -----------------------------
    def _call_llm(self, state: dict) -> str:
        try:
            prompt = f"""
You are optimizing LNG supply.

Storage: {state['storage']['level']}
Demand: {state['demand_forecast'][state['time_step']]}

Suggest: wait / store / hedge / release_25 / release_50
"""

            res = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

            return res.text.strip().lower()
        except:
            return "wait"

    def _parse_llm(self, text: str):
        if "store" in text:
            return {"type": "store", "parameters": {"amount": 20}}
        if "hedge" in text:
            return {"type": "hedge", "parameters": {}}
        if "50" in text:
            return {"type": "release", "parameters": {"amount": 50}}
        if "25" in text:
            return {"type": "release", "parameters": {"amount": 25}}
        return {"type": "wait", "parameters": {}}

    # -----------------------------
    # FINAL DECISION
    # -----------------------------
    def act(self, state: dict) -> dict:
        baseline = self._baseline_action(state)

        # LLM only as refinement (safe)
        llm_text = self._call_llm(state)
        llm_action = self._parse_llm(llm_text)

        # simulate both
        base_sim = self._simulate_step(state, baseline)
        llm_sim = self._simulate_step(state, llm_action)

        # choose lower deficit
        if llm_sim["deficit"] < base_sim["deficit"]:
            return llm_action

        return baseline
    

# agent.py v2
# the following code calls the llm for every step (if we have no rate limit optimal code better results)
'''

from google import genai

class LNGAgent:
    def __init__(self, model_name: str, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    # -----------------------------
    # 🔥 STRONG PROMPT (V2)
    # -----------------------------
    def _build_prompt(self, state):
        t = state["time_step"]

        return f"""
You are an expert LNG supply optimizer.

CRITICAL:
- Release REDUCES storage
- Store/Hedge INCREASE supply
- Avoid shortage (highest priority)

STATE:
Demand: {state['demand_forecast'][t]}
Storage: {state['storage']['level']}/{state['storage']['capacity']}
Budget: {state['budget']}
Ships: {state['ships']}
Blocked Routes: {state['blocked_routes']}

Think step-by-step:
1. Will shortage occur?
2. Do we need more supply?
3. Is storage too high?

Choose ONE action.

ACTIONS:
wait, store, hedge, release_25, release_50, reroute

ONLY OUTPUT ACTION.
"""

    # -----------------------------
    # LLM CALL
    # -----------------------------
    def _call_llm(self, state):
        try:
            prompt = self._build_prompt(state)

            res = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

            return res.text.strip().lower()
        except:
            return "wait"

    # -----------------------------
    # PARSER
    # -----------------------------
    def _parse(self, text):
        if "store" in text:
            return {"type": "store", "parameters": {"amount": 20}}
        if "hedge" in text:
            return {"type": "hedge", "parameters": {}}
        if "reroute" in text:
            return {"type": "reroute", "parameters": {"ship_id": 1, "new_route": "Atlantic"}}
        if "50" in text:
            return {"type": "release", "parameters": {"amount": 50}}
        if "25" in text:
            return {"type": "release", "parameters": {"amount": 25}}
        return {"type": "wait", "parameters": {}}

    # -----------------------------
    # SIMULATION (safety)
    # -----------------------------
    def _simulate(self, state, action):
        storage = state["storage"]["level"]
        capacity = state["storage"]["capacity"]

        if action["type"] == "release":
            storage = max(0, storage - action["parameters"].get("amount", 0))

        elif action["type"] == "store":
            storage = min(capacity, storage + action["parameters"].get("amount", 0))

        elif action["type"] == "hedge":
            storage += 20

        t = state["time_step"]
        demand = state["demand_forecast"][t]

        ships = state.get("ships", [])
        incoming = sum(s["capacity"] for s in ships if s.get("eta", 999) <= 1)

        supply = storage + incoming
        deficit = max(0, demand - supply)

        return deficit

    # -----------------------------
    # FALLBACK (safe heuristic)
    # -----------------------------
    def _fallback(self, state):
        t = state["time_step"]
        demand = state["demand_forecast"][t]
        storage = state["storage"]["level"]

        ships = state.get("ships", [])
        incoming = sum(s["capacity"] for s in ships if s.get("eta", 999) <= 1)

        if demand > storage + incoming:
            return {"type": "store", "parameters": {"amount": 20}}

        return {"type": "wait", "parameters": {}}

    # -----------------------------
    # FINAL ACT
    # -----------------------------
    def act(self, state):
        llm_text = self._call_llm(state)
        llm_action = self._parse(llm_text)

        fallback = self._fallback(state)

        # simulate both
        llm_deficit = self._simulate(state, llm_action)
        fb_deficit = self._simulate(state, fallback)

        # pick safer
        if llm_deficit <= fb_deficit:
            return llm_action

        return fallback

'''