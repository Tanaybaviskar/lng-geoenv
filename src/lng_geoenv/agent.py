
class LNGAgent:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name
        self.cache = {}  # 🔥 state-action cache

    # -----------------------------
    # 🔥 STATE SIGNATURE (for caching)
    # -----------------------------
    def _state_key(self, state):
        t = state["time_step"]
        demand = int(state["demand_forecast"][t] // 10)
        storage = int(state["storage"]["level"] // 10)
        return (t, demand, storage)

    # -----------------------------
    # 🔥 LLM TRIGGER (VERY IMPORTANT)
    # -----------------------------
    def should_call_llm(self, state):
        t = state["time_step"]
        demand = state["demand_forecast"][t]
        storage = state["storage"]["level"]

        ships = state.get("ships", [])
        blocked = state.get("blocked_routes", [])

        incoming = sum(s["capacity"] for s in ships if s.get("eta", 999) <= 1)
        supply = storage + incoming
        deficit = demand - supply

        if deficit > 20:
            return True
        if len(blocked) > 0:
            return True
        if t % 5 == 0:
            return True

        return False

    # -----------------------------
    # 🔥 BASELINE (fast fallback)
    # -----------------------------
    def baseline(self, state):
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

        if storage > 0.85 * capacity:
            return {"type": "release", "parameters": {"amount": 20}}

        return {"type": "wait", "parameters": {}}

    # -----------------------------
    # 🔥 LLM CALL (OpenAI format)
    # -----------------------------
    def call_llm(self, prompt):
        try:
            #print("🔥 LLM CALL START")

            response = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                max_output_tokens=10,
                temperature=0.0,
            )

            # 🔥 SAFE EXTRACTION
            text = ""

            if hasattr(response, "output") and response.output:
                for item in response.output:
                    if hasattr(item, "content"):
                        for c in item.content:
                            if hasattr(c, "text") and c.text:
                                text += c.text

            text = text.strip().lower()

            if not text:
                print("⚠️ EMPTY RESPONSE → fallback to wait")
                return "wait"

            # print("✅ LLM RAW:", text)

            return text

        except Exception as e:
            print("❌ LLM ERROR:", e)
            return "wait"



    # -----------------------------
    # 🔥 PARSE
    # -----------------------------
    VALID_ACTIONS = {
        "wait": {"type": "wait", "parameters": {}},
        "store": {"type": "store", "parameters": {"amount": 20}},
        "hedge": {"type": "hedge", "parameters": {}},
        "release_20": {"type": "release", "parameters": {"amount": 20}},
        "release_50": {"type": "release", "parameters": {"amount": 50}},
        "reroute": {"type": "reroute", "parameters": {"ship_id": 1, "new_route": "Atlantic"}},
    }

    def parse(self, text):
        text = (text or "").strip().lower()

        for key in self.VALID_ACTIONS:
            if key in text:
                return self.VALID_ACTIONS[key]

        # 🔥 fallback to baseline instead of wait
        return None


    # -----------------------------
    # 🔥 SAFETY FILTER
    # -----------------------------
    def safe(self, state, action):
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

        return action

    # -----------------------------
    # 🔥 FINAL DECISION
    # -----------------------------
    def act(self, state):
        key = self._state_key(state)

        # ✅ cache hit
        if key in self.cache:
            return self.cache[key]

        # ✅ decide whether to call LLM
        if self.should_call_llm(state):
            text = self.call_llm(self._build_prompt(state))
            action = self.parse(text)

            if action is None:
                action = self.baseline(state)
        else:
            action = self.baseline(state)

        action = self.safe(state, action)

        # ✅ store in cache
        self.cache[key] = action

        return action

    # -----------------------------
    # PROMPT
    # -----------------------------
    def _build_prompt(self, state):
        t = state["time_step"]

        return f"""
    You must choose ONE action.

    Allowed actions:
    wait
    store
    hedge
    release_20
    release_50
    reroute

    Rules:
    - Output ONLY one word from the list
    - Do NOT explain
    - Do NOT write sentences
    - If you output anything else, it is WRONG

    State:
    Demand: {state['demand_forecast'][t]}
    Storage: {state['storage']['level']}
    Blocked routes: {state.get('blocked_routes', [])}

    Answer:
    """
