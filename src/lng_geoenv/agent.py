import json
import logging
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import time

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class GeminiAgentConfig:
    """Configuration for Gemini API and agent behavior."""

    def __init__(self):
        self.api_key = self._get_api_key()
        self.model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
        self.temperature = float(os.getenv("AGENT_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("AGENT_MAX_TOKENS", "500"))
        self.timeout = int(os.getenv("AGENT_TIMEOUT", "30"))

    def _get_api_key(self) -> str:
        """
        Retrieve API key from environment variables with fallback.
        Priority: GEMINI_API_KEY -> HF_TOKEN
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            api_key = os.getenv("HF_TOKEN")
        if not api_key:
            raise ValueError(
                "API key not found. Set GEMINI_API_KEY or HF_TOKEN environment variable."
            )
        return api_key

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return bool(self.api_key) and GEMINI_AVAILABLE


class GeminiAgent:
    """
    Intelligent LNG Crisis Agent powered by Google Gemini.

    Responsibilities:
    - Analyze LNG supply/demand state
    - Generate strategic actions
    - Ensure safe action execution
    - Provide explainable decisions
    """

    VALID_ACTIONS = ["wait", "store", "release", "reroute", "hedge"]

    def __init__(self, use_llm: bool = True):
        """
        Initialize the Gemini Agent.

        Args:
            use_llm: If False, use baseline policy (useful for testing)
        """
        self.use_llm = use_llm and GEMINI_AVAILABLE
        self.config = None
        self.model = None
        self.action_history: List[Dict] = []
        self.previous_action: Optional[Dict] = None

        # Rate limiting tracking
        self.request_times: List[datetime] = []  # For RPM tracking
        self.daily_request_count = 0  # For RPD tracking
        self.daily_reset_time = datetime.utcnow()

        if self.use_llm:
            try:
                self.config = GeminiAgentConfig()
                genai.configure(api_key=self.config.api_key)
                self.model = genai.GenerativeModel(self.config.model_name)
                logger.info(
                    f"✓ Gemini Agent initialized with model: {self.config.model_name}"
                )
            except Exception as e:
                logger.warning(
                    f"✗ Failed to initialize Gemini: {e}. Using baseline policy."
                )
                self.use_llm = False
        else:
            logger.info("→ Using baseline policy (LLM disabled)")

    def format_state_for_llm(self, state: Dict[str, Any]) -> str:
        """
        Convert raw state dictionary into human-readable format for LLM.
        This is CRITICAL for LLM understanding.

        Args:
            state: Current environment state

        Returns:
            Formatted string for LLM prompt
        """
        t = state.get("time_step", 0)
        storage = state.get("storage", {})

        # Format ships
        ships_info = ""
        for ship in state.get("ships", []):
            ships_info += (
                f"  - Ship {ship['id']}: {ship['origin']} → {ship['destination']} "
            )
            ships_info += (
                f"(ETA: {ship['eta']} steps, Capacity: {ship['capacity']}LNG, "
            )
            ships_info += f"Route: {ship['route']}, Status: {ship['status']})\n"

        # Format blocked routes
        blocked = ", ".join(state.get("blocked_routes", []))

        # Calculate supply/demand metrics
        current_demand = state.get("demand", 0)
        incoming_supply = sum(
            ship["capacity"]
            for ship in state.get("ships", [])
            if ship.get("eta", 0) <= 2 and ship.get("status") == "moving"
        )
        current_storage = storage.get("level", 0)
        total_supply = current_storage + incoming_supply

        shortage_alert = "⚠️ " if total_supply < current_demand else "✓ "

        state_text = f"""
=== LNG CRISIS MANAGEMENT STATE ===
Time Step: {t}
Current Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

STORAGE STATUS:
  Current Level: {current_storage:.1f} / {storage.get("capacity", 100):.1f} LNG
  Utilization: {(current_storage / storage.get("capacity", 100) * 100):.1f}%

DEMAND & SUPPLY:
  Current Demand: {current_demand:.1f} LNG
  Available Supply: {total_supply:.1f} LNG (storage + incoming)
  {shortage_alert}Shortage Risk: {max(0, current_demand - total_supply):.1f} LNG

FINANCIAL:
  Budget: ${state.get("budget", 0):.2f}
  Current Price: ${state.get("price", 0):.2f} per LNG
  
INCOMING SHIPS (Next 7 steps):
{ships_info if ships_info else "  No incoming ships"}

BLOCKED ROUTES:
  {blocked if blocked else "All routes operational"}

DECISION CONTEXT:
1. Avoid LNG shortage → critical priority
2. Minimize operational costs (fuel, storage, hedging)
3. Manage budget efficiently
4. Avoid unnecessary reroutes
5. Plan ahead for demand spikes"""

        return state_text

    def _build_prompt(self, state_text: str) -> str:
        """
        Build the prompt for LLM.

        Args:
            state_text: Formatted state information

        Returns:
            Complete prompt for LLM
        """
        base_prompt = f"""You are an expert LNG supply chain crisis manager.
Your goal: Make strategic decisions to prevent shortages while minimizing costs.

VALID ACTIONS (pick ONE):
1. "wait" - Do nothing, monitor situation
2. "store" - Buy LNG on spot market and store (costs money, adds supply)
3. "release" - Release stored LNG to meet demand immediately
4. "reroute" - Reroute a ship to a different route (adds delay, avoids blockages)
5. "hedge" - Buy options/futures to protect against price spikes (costs $10)

{state_text}

{"=" * 60}"""

        base_prompt += """

THINKING STRUCTURE
Think step-by-step before responding:

1. CHECK CURRENT SITUATION
   - Is storage above 80%? → Consider release
   - Is storage below 30%? → Consider store or hedge
   - Is current_demand > available_supply? → SHORTAGE ALERT
   
2. CHECK BLOCKED ROUTES
   - Are critical routes blocked?
   - How many ships are delayed?
   - Can rerouting help?
   
3. EVALUATE BUDGET
   - Do we have budget for store/hedge?
   - Is price favorable for buying?
   
4. MAKE DECISION
   - Pick action with best risk-adjusted return
   - Avoid repeated actions
   - Ensure budget enforcement

CONSIDER FUTURE
Think about next 2 steps:
- If I store now, will demand in 2 steps require it?
- Will any ships arrive soon, reducing need to store?
- What's the cheapest time to buy?
- Can I delay action 1 step for better conditions?
"""

        base_prompt += """

RESPOND ONLY WITH VALID JSON (no other text):
{
  "reasoning": "Brief explanation of your decision",
  "action_type": "<one of: wait, store, release, reroute, hedge>",
  "parameters": {
    "amount": <if store/release: LNG amount, 0-100>,
    "ship_id": <if reroute: which ship (1 or 2)>,
    "new_route": <if reroute: Suez|Panama|Atlantic|Hormuz>
  },
  "confidence": <0.0-1.0>,
  "warnings": "<any concerns or edge cases>"
}"""

        return base_prompt

    def _check_rate_limits(self) -> bool:
        """
        Check and enforce rate limits (RPM, TPM, RPD).

        Returns:
            True if within limits, False if rate limited
        """
        from src.lng_geoenv.config import Config

        now = datetime.utcnow()

        # Check daily limit (RPD)
        if now - self.daily_reset_time > timedelta(days=1):
            self.daily_reset_time = now
            self.daily_request_count = 0

        rpd_limit = Config.get_gemini_rpd()
        if self.daily_request_count >= rpd_limit:
            logger.warning(
                f"✗ Daily request limit reached: {self.daily_request_count}/{rpd_limit}"
            )
            return False

        # Check per-minute limit (RPM)
        rpm_limit = Config.get_gemini_rpm()
        one_minute_ago = now - timedelta(minutes=1)
        self.request_times = [t for t in self.request_times if t > one_minute_ago]

        if len(self.request_times) >= rpm_limit:
            sleep_time = (self.request_times[0] - one_minute_ago).total_seconds() + 1
            logger.warning(
                f"✗ Rate limited: RPM {len(self.request_times)}/{rpm_limit}. Waiting {sleep_time:.1f}s"
            )
            time.sleep(sleep_time)

        self.request_times.append(now)
        self.daily_request_count += 1
        return True

    def _call_gemini(self, prompt: str) -> Optional[str]:
        """
        Call Gemini API with proper error handling and timeouts.

        Args:
            prompt: The prompt to send to Gemini

        Returns:
            Response text or None if failed
        """
        if not self.model:
            return None

        # Check rate limits before calling API
        if not self._check_rate_limits():
            logger.warning("Rate limit exceeded, using baseline policy")
            return None

        try:
            logger.debug("Sending request to Gemini...")
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=4096,  # Increased to maximum allowed
                ),
                safety_settings={
                    "HARASSMENT": "BLOCK_NONE",
                    "HATE_SPEECH": "BLOCK_NONE",
                    "SEXUAL": "BLOCK_NONE",
                    "DANGEROUS": "BLOCK_NONE",
                },
            )

            if response and response.text:
                logger.info(
                    f"✓ Gemini response ({len(response.text)} chars): {response.text[:150]}..."
                )
                return response.text
            else:
                logger.warning(f"Empty response from Gemini: {response}")
                return None

        except Exception as e:
            logger.error(f"Gemini API error: {type(e).__name__}: {e}")
            return None

    # Parse and validate action

    def _parse_action(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Safely parse JSON response from LLM.

        Args:
            response_text: Raw text response from LLM

        Returns:
            Parsed action dict or None if parsing failed
        """
        # Debug: Write full response to file for inspection
        with open("/tmp/gemini_response.txt", "w") as f:
            f.write(f"Response type: {type(response_text)}\n")
            f.write(f"Response length: {len(response_text)}\n")
            f.write(f"Response content:\n{response_text}\n")

        # Step 1: Aggressively strip markdown code blocks
        text_to_parse = response_text.strip()

        # Remove ```json prefix
        if text_to_parse.startswith("```json"):
            text_to_parse = text_to_parse[7:]  # Remove "```json"
        elif text_to_parse.startswith("```"):
            text_to_parse = text_to_parse[3:]  # Remove "```"

        # Remove trailing ```
        if text_to_parse.endswith("```"):
            text_to_parse = text_to_parse[:-3]

        text_to_parse = text_to_parse.strip()

        # Step 2: Try direct JSON parsing
        try:
            action = json.loads(text_to_parse)
            logger.info(f"✓ Parsed action: {action.get('action_type', 'unknown')}")
            return action
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")

        # Step 3: Extract JSON from response using bracket search
        try:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                action = json.loads(json_str)
                logger.info(f"✓ Extracted JSON: {action.get('action_type', 'unknown')}")
                return action
        except Exception as e:
            logger.warning(f"Extraction error: {e}")

        logger.warning(
            f"✗ Could not parse response (first 150 chars): {response_text[:150]}..."
        )
        return None

    def _validate_action(
        self, action: Dict[str, Any], state: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Validate action against guardrails and state constraints.

        Guardrails implementation

        Args:
            action: Parsed action from LLM
            state: Current environment state

        Returns:
            (is_valid, reason)
        """
        action_type = action.get("action_type", "").lower()

        # Check 1: Valid action type
        if action_type not in self.VALID_ACTIONS:
            return False, f"Invalid action type: {action_type}"

        # Check 2: Prevent repeated reroutes (guardrail)
        if action_type == "reroute":
            if (
                self.previous_action
                and self.previous_action.get("action_type") == "reroute"
            ):
                return False, "Cannot reroute consecutive steps (guardrail)"

            recent_reroutes = sum(
                1 for a in self.action_history[-3:] if a.get("action_type") == "reroute"
            )
            if recent_reroutes >= 2:
                return False, "Too many reroutes recently (guardrail)"

        # Check 3: Budget enforcement
        if action_type == "hedge":
            if state.get("budget", 0) < 10:
                return False, "Insufficient budget for hedge"

        # Check 4: Storage bounds for store/release
        if action_type == "store":
            amount = action.get("parameters", {}).get("amount", 0)
            storage = state.get("storage", {})
            available_space = storage.get("capacity", 100) - storage.get("level", 0)
            if amount > available_space:
                action["parameters"]["amount"] = available_space
                return (
                    True,
                    f"Clamped store amount to available capacity: {available_space}",
                )

        elif action_type == "release":
            amount = action.get("parameters", {}).get("amount", 0)
            storage = state.get("storage", {})
            if amount > storage.get("level", 0):
                action["parameters"]["amount"] = storage.get("level", 0)
                return (
                    True,
                    f"Clamped release amount to current storage: {storage.get('level', 0)}",
                )

        # Check 5: Ship validation for reroute
        if action_type == "reroute":
            ship_id = action.get("parameters", {}).get("ship_id")
            ships = {ship["id"]: ship for ship in state.get("ships", [])}
            if ship_id not in ships:
                return False, f"Ship {ship_id} not found"

        return True, "Valid"

    def _baseline_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback policy when LLM fails or is disabled.
        Simple rule-based logic.

        Args:
            state: Current environment state

        Returns:
            Action dict
        """
        storage = state.get("storage", {})
        level = storage.get("level", 0)
        capacity = storage.get("capacity", 100)
        demand = state.get("demand", 0)
        budget = state.get("budget", 0)

        # Rule 1: If storage critically low and demand high, release what we have
        if level < 10 and demand > 50:
            return {
                "action_type": "release",
                "parameters": {"amount": min(20, level)},
                "confidence": 0.6,
                "reasoning": "Baseline: Critical shortage, releasing storage",
            }

        # Rule 2: If storage low and budget available, store
        if level < 50 and budget >= 30:
            return {
                "action_type": "store",
                "parameters": {"amount": 30},
                "confidence": 0.7,
                "reasoning": "Baseline: Low storage, buying to ensure supply",
            }

        # Rule 3: If storage dangerously high, release to save costs
        if level > 150:
            return {
                "action_type": "release",
                "parameters": {"amount": 40},
                "confidence": 0.5,
                "reasoning": "Baseline: High storage, reducing holding costs",
            }

        # Default: Wait and observe
        return {
            "action_type": "wait",
            "parameters": {},
            "confidence": 0.5,
            "reasoning": "Baseline: Monitoring situation, no immediate action needed",
        }

    def _simulate_step(
        self, action: Dict[str, Any], state: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Quick simulation of an action to evaluate its impact.

        Lookahead simulation

        Args:
            action: Action to simulate
            state: Current state

        Returns:
            (projected_reward, projected_state)
        """
        # Simple projection: check if action helps with shortage
        storage = state.get("storage", {}).copy()
        demand = state.get("demand", 0)
        budget = state.get("budget", 0)

        action_type = action.get("action_type", "wait")
        params = action.get("parameters", {})

        projected_shortage = max(0, demand - storage.get("level", 0))

        if action_type == "store":
            amount = params.get("amount", 0)
            storage["level"] = min(
                storage.get("capacity", 100), storage.get("level", 0) + amount
            )
            projected_shortage = max(0, demand - storage["level"])
            projected_cost = -amount * state.get("price", 100)

        elif action_type == "release":
            amount = params.get("amount", 0)
            storage["level"] = max(0, storage.get("level", 0) - amount)
            projected_shortage = max(0, demand - storage["level"])
            projected_cost = 0

        elif action_type == "wait":
            projected_shortage = max(0, demand - storage.get("level", 0))
            projected_cost = 0

        elif action_type == "hedge":
            projected_shortage = projected_shortage * 0.9  # Slight hedge benefit
            projected_cost = -10

        else:
            projected_cost = -5

        # Simple reward proxy
        reward = projected_cost - (projected_shortage * 5)

        return reward, {"storage": storage, "projected_shortage": projected_shortage}

    def _lookahead_analysis(
        self, candidate_actions: List[Dict], state: Dict[str, Any]
    ) -> Optional[Dict]:
        """
        Compare top 2-3 candidate actions using 1-step lookahead.

        Args:
            candidate_actions: List of actions to evaluate
            state: Current state

        Returns:
            Best action after lookahead analysis
        """
        if not candidate_actions:
            return None

        best_action = candidate_actions[0]
        best_score = float("-inf")

        for action in candidate_actions[:3]:  # Evaluate top 3
            score, _ = self._simulate_step(action, state)
            logger.debug(
                f"Lookahead score for {action.get('action_type')}: {score:.2f}"
            )

            if score > best_score:
                best_score = score
                best_action = action

        logger.info(
            f"✓ Lookahead selected: {best_action.get('action_type')} (score: {best_score:.2f})"
        )
        return best_action

    def choose_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main decision-making function.

        Steps:
        1. Format state for LLM
        2. Call Gemini API
        3. Parse response
        4. Validate with guardrails
        5. Apply fallback if needed
        6. Log and return

        Args:
            state: Current environment state

        Returns:
            Action dict ready for env.step() with format:
            {
                "type": "action_type",  # for env.step()
                "parameters": {...},
                ... additional metadata fields
            }
        """

        state_text = self.format_state_for_llm(state)
        prompt = self._build_prompt(state_text)

        if not self.use_llm:
            logger.info("→ Using baseline policy (LLM disabled)")
            action = self._baseline_action(state)
        else:
            response_text = self._call_gemini(prompt)

            if not response_text:
                logger.warning("✗ Gemini call failed, using baseline")
                action = self._baseline_action(state)
            else:
                parsed = self._parse_action(response_text)

                if not parsed:
                    logger.warning("✗ Could not parse LLM response, using baseline")
                    action = self._baseline_action(state)
                else:
                    action = parsed

                    is_valid, reason = self._validate_action(action, state)
                    if not is_valid:
                        logger.warning(f"✗ Action failed validation: {reason}")
                        action = self._baseline_action(state)
                    else:
                        logger.info(f"✓ Action validated: {reason}")

                        # Lookahead optimization
                        alternatives = [
                            action,
                            {"action_type": "wait", "parameters": {}},
                            {"action_type": "store", "parameters": {"amount": 30}},
                        ]
                        best = self._lookahead_analysis(alternatives, state)
                        if best:
                            action = best

        # Ensure action has all required fields
        if "reasoning" not in action:
            action["reasoning"] = "No reasoning provided"
        if "confidence" not in action:
            action["confidence"] = 0.5

        # Normalize: convert "action_type" to "type" for compatibility with env.step()
        if "action_type" in action and "type" not in action:
            action["type"] = action["action_type"]

        # Track action history
        self.action_history.append(action)
        self.previous_action = action

        logger.info(
            f"→ Selected action: {action.get('type', action.get('action_type', 'unknown'))} (confidence: {action.get('confidence', 0):.2f})"
        )

        return action
