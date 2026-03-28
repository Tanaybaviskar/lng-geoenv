# LNG-GeoEnv: LNG Supply Chain Simulation under Geopolitical Disruptions

**Engineering decisions that keep the world running when supply chains break.**

---

## 1. Project Title

LNG-GeoEnv: A reinforcement learning environment simulating real-world Liquefied Natural Gas (LNG) supply chain optimization under geopolitical disruptions.

---

## 2. Overview

LNG is critical infrastructure—a single route blockade can cascade through global energy markets. This environment simulates the decision-making problem faced by logistics coordinators during supply chain crises.

An **agent** observes a dynamic LNG supply system and sequentially decides how to:
- Allocate limited storage
- Reroute vessels around blocked corridors
- Hedge price risk
- Manage demand volatility

The environment provides **continuous reward signals** that reflect the trade-off between supply security, operational cost, and stability. Episodes are deterministic and reproducible.

### Real-World Context

- **February 2022**: Ever Given blockage in Suez Canal disrupted 12% of global trade for 6 days
- **2024 Red Sea Crisis**: Houthi attacks forced rerouting via Cape of Good Hope, adding 10–15 days per journey
- **Energy Security**: European LNG terminals operate near capacity; shortages trigger price spikes ($40/MMBtu → historic peaks)

This simulation models the **decision-making problem** during such crises: given imperfect demand forecasts, limited budget, and dynamic blockages, what actions minimize shortage risk while managing cost?

---

## 3. Problem Motivation

### The Challenge

LNG supply chains face three simultaneous pressures:

1. **Demand Variability**: Weather, industrial demand, and geopolitical shocks create unpredictable consumption patterns (AR(1) process with occasional shocks up to ±50 units)
2. **Route Disruptions**: Dominant corridors (Suez, Panama, Hormuz) are geopolitically fragile; blockages force expensive detours (+2 weeks ETA penalty)
3. **Constrained Resources**: Importing regions have limited storage (200-unit capacity); hedging is expensive (cost: 10 per use); budget is finite (500 units)

### Why This Matters

Agents must learn to **make trade-offs**:
- Release inventory now vs. risk shortage later?
- Reroute early vs. absorb potential blockage?
- Hedge expensive protection vs. take price risk?

No single optimal policy exists—decisions depend on forecasted demand, price, risk, and available budget. This mirrors real operator decision-making.

---

## 4. Environment Design

### Simulation Architecture

```
Agent → choose_action(state, demand) → action
         ↓
         env.step(action)
         ↓
       update world (ship movement, arrivals, blockages)
         ↓
       compute demand (AR(1) + shocks + seasonality)
         ↓
       apply reward (cost + shortage + delay + risk penalties)
         ↓
       normalize reward → [0, 1]
         ↓
       return (new_state, reward, done, info)
```

### State Dynamics

**Ships evolve** as follows:
- Each timestep, all moving ships have ETA decremented by 1
- If ETA reaches 0, ship transitions to "arrived" and cargo transfers to storage (clamped by capacity)
- Blockages delay ships by +1 ETA per step

**Demand evolves** via AR(1) process:
$$d_t = \phi \cdot d_{t-1} + (1 - \phi) \cdot \mu + \epsilon_t + \text{shock}_t + \text{seasonal}_t$$

where:
- $\phi = 0.7$ (persistence)
- $\mu = 100$ (base demand)
- $\epsilon_t \sim N(0, \sigma^2)$ (noise)
- Shocks: 5–15% probability per episode
- Seasonality: amplitude varies by task

**Storage and price** respond to actions and system state; budget decreases with hedge operations.

### Episode Structure

- Each episode runs for **10 steps**
- Deterministic seeding (seed=42 for all tasks)
- Episode ends when max_steps reached or explicit termination triggered
- Final score is **averaged normalized reward** across all steps, clipped to [0.0, 1.0]

---

## 5. Observation Space

Agents observe the full state at each timestep:

```python
observation = {
    "time_step": int,                   # Current step [0, 9]
    "ships": [                          # List of in-transit and arrived vessels
        {
            "id": int,                  # Ship identifier
            "origin": str,              # Origin port (e.g., "Qatar")
            "destination": str,         # Destination (e.g., "Europe")
            "eta": int,                 # Steps until arrival
            "capacity": float,          # Cargo capacity (units)
            "route": str,               # Current route (Suez|Panama|Atlantic|Hormuz)
            "status": str               # "moving" | "arrived" | "done"
        }
    ],
    "blocked_routes": [str],            # Routes currently blocked (subset of known routes)
    "storage": {
        "level": float,                 # Current stored LNG (units)
        "capacity": float               # Max storage (200 units)
    },
    "demand": float,                    # Current timestep demand (units)
    "price": float,                     # Spot LNG price ($/unit) [50, 150]
    "budget": float                     # Remaining capital for hedging
}
```

**Key properties:**
- Full observability (agents see complete state)
- All values are deterministic given seed and action history
- Approximately 20 scalar features total

---

## 6. Action Space

Agents take one action per timestep. Format:

```python
action = {
    "type": str,                        # Action category
    "parameters": dict                  # Action-specific parameters
}
```

### Supported Actions

| Action | Parameters | Effect | Cost |
|--------|-----------|--------|------|
| **wait** | `{}` | No operation; advance time | 0 |
| **release** | `{"amount": float}` | Release stored LNG to market; meets demand | 0 |
| **reroute** | `{"ship_id": int, "new_route": str}` | Redirect ship to new corridor | +2 ETA |
| **hedge** | `{}` | Financial protection; increased supply buffer | 10 budget |
| **store** | `{"amount": float}` | Increase storage (if budget allows) | amount × price |

### Constraints

- **release**: Cannot exceed current storage level
- **reroute**: New route must differ from current; only works if ship is "moving"
- **hedging**: Costs 10 budget units; adds +20 supply as safety margin
- **store**: Blocked by insufficient budget or storage capacity

---

## 7. Tasks

Three difficulty tiers with deterministic configuration changes:

### Task 1: Stable (Easy)

**Scenario**: Favorable LNG market with predictable demand and minimal geopolitical risk.

**Configuration:**
- Demand volatility: Low (σ=10, shock probability=0.05)
- Route risk: Minimal (risk_scale=0.2)
- Seasonality: Weak (amplitude=5)
- Blocked routes: ≤1 at any time

**Baseline strategy**: Hold 80% storage; hedge prophylactically; occasional rerouting.

**Expected agent performance**: Agents can maintain stable supply with simple policies. Maximum shortage is rare.

---

### Task 2: Volatile (Medium)

**Scenario**: Active LNG market with price fluctuations, supply shocks, and intermittent route disruptions.

**Configuration:**
- Demand volatility: Moderate (σ=10, shock probability=0.15)
- Route risk: Medium (risk_scale=0.5)
- Seasonality: Moderate (amplitude=10)
- Blocked routes: 1–2 intermittently

**Baseline strategy**: Adaptive rerouting; hedge on price spikes (price > 120); dynamic storage management.

**Expected agent performance**: Agents must learn to balance hedging costs against shortage risk. Simple policies fail; reactive decision-making required.

---

### Task 3: War (Hard)

**Scenario**: Supply chain crisis with multiple simultaneous disruptions, extreme demand volatility, and severe funding constraints.

**Configuration:**
- Demand volatility: High (σ=10, shock probability=0.3)
- Route risk: Critical (risk_scale=0.9)
- Seasonality: Strong (amplitude=15)
- Blocked routes: 1–3 frequently

**Baseline strategy**: Aggressive rerouting; continuous hedging; minimal storage holding; risk-aware fuel cost management.

**Expected agent performance**: Agents face genuine trade-off conflicts. High-performing policies require forward planning and multi-step lookahead.

---

## 8. Reward Design

### Objective Function

Reward is a weighted sum of penalties for undesirable outcomes, normalized to [0, 1]:

**Raw penalty:**
$$\text{penalty} = w_c \cdot C + w_s \cdot S + w_d \cdot D + w_r \cdot R$$

**Components:**
- $C$ = Transport cost (fuel per nautical mile) + Storage holding cost (0.02 per unit)
- $S$ = Shortage penalty (quadratic: $\alpha \cdot d^2$ where $d$ is deficit, strongly penalizes unmet demand)
- $D$ = Delay penalty ($\beta \cdot \sum_{i} \text{ETA}_{i}$, discourages slow routes)
- $R$ = Risk penalty ($\gamma \cdot r \cdot v$ where $r$ is risk and $v$ is cargo value, penalizes carrying high-value cargo on risky routes)

**Normalized reward:**
$$r_t = \text{normalize}(-\text{penalty}_t) \in [0, 1]$$

where the `RewardNormalizer` tracks running min/max and rescales to [0, 1] online.

### Reward Coefficients (per-task configurable)

**All tasks use:**
- $w_c = 1.0$ (cost weight)
- $w_s = 6.0$ (shortage heavily penalized)
- $w_d = 1.0$ (delay weight)
- $w_r = 3.0$ (risk weight)
- $\alpha = 2.0$ (shortage quadratic exponent)
- $\beta = 1.0$ (delay multiplier)
- $\gamma = 2.0$ (risk multiplier)

### Scoring

Episode score is the **average normalized reward** across all steps:
$$\text{score} = \frac{1}{N} \sum_{t=1}^{N} r_t, \quad \text{clipped to } [0, 1]$$

This ensures:
- Agents are rewarded for consistent good decisions, not just end-episode outcomes
- Partial progress is visible
- Final score is always in [0.0, 1.0]

---

## 9. Baseline Agent

The baseline policy in `choose_action()` implements a prioritized decision tree designed for transparency and debuggability:

```python
if storage_level < demand:
    # Shortage imminent → reroute blocked ships
    for each ship on blockaded route:
        reroute to available corridor
    # If no reroutable ship, release emergency reserves
    release min(shortage_amount * 0.8, storage_level)

# Proactive rerouting
for each ship on blocked route:
    reroute to available corridor

# Hedging on price spikes
if price > 120 and budget >= 10:
    hedge()

# Storage balancing
if storage_ratio > 0.85:
    release (storage_level - 0.7 * capacity) * 0.5

# Otherwise
wait()
```

**Philosophy**: Simple, interpretable, non-learning. Demonstrates that the environment is solvable with basic heuristics but benefits from learned policies.

**Reproducible scores** (on seed=42):
- Stable: ≈0.59
- Volatile: ≈0.58
- War: ≈0.66

---

## 10. Execution Flow

### Episode Lifecycle

```
reset(seed=42)
  └─ Initialize ships with deterministic ETA
  └─ Sample demand_gen with seed
  └─ Set storage to initial level
  └─ Return initial_state

for step in range(max_steps):
  state = env.get_state()
  demand = state["demand"]
  
  action = agent.choose_action(state, demand)
  
  (new_state, reward, done, info) = env.step(action)
  
  history.append({
    "state": new_state,
    "action": action,
    "reward": reward  # Already normalized [0, 1]
  })
  
  if done or step == max_steps - 1:
    break

evaluation = evaluate_episode(history)
  └─ Compute average normalized reward
  └─ Clip to [0, 1]
  └─ Return final_score
```

### Determinism

All randomness uses `np.random.seed(seed)` set at environment reset. Given fixed seed and action sequence, output is deterministic.

---

## 11. Setup Instructions

### Local Execution

**Requirements:** Python ≥ 3.12

**Installation:**

```bash
# Clone repository
git clone https://github.com/<org>/lng-geoenv.git
cd lng-geoenv

# Install dependencies (using uv, recommended)
uv sync
```

**Run baseline inference:**

```bash
python inference.py
```

**Output:**
- Per-task execution logs (storage, demand, action, reward)
- Final scores for all 3 tasks
- Aggregate statistics

**Run tests:**

```bash
pytest tests/
```

Validates environment correctness, reward bounds, storage constraints, and grader logic.

### Docker Execution

**Build:**

```bash
docker build -t lng-geoenv .
```

**Run:**

```bash
docker run lng-geoenv
```

---

## 12. Project Structure

```
lng-geoenv/
├── src/lng_geoenv/
│   ├── __init__.py
│   ├── env.py                      # LNGEnv class, step() / reset()
│   ├── world.py                    # Ship dynamics, route risk
│   ├── demand.py                   # DemandGenerator (AR(1) + shocks)
│   ├── reward.py                   # RewardEngine, normalization
│   ├── grader.py                   # EpisodeGrader, scoring
│   ├── models.py                   # TypedDicts: Ship, Storage, State
│   └── tasks.py                    # get_task_config(task_name)
├── tests/
│   ├── test_env.py                 # Environment correctness tests
│   ├── test_demand.py              # Demand generator tests
│   └── test_reward.py              # Reward logic tests
├── inference.py                    # Baseline agent execution pipeline
├── main.py                         # Debug script
├── openenv.yaml                    # OpenEnv specification
├── Dockerfile                      # Container configuration
├── pyproject.toml                  # Dependencies
├── README.md                       # This file
└── LICENSE                         # License
```

---

## 13. Compliance with OpenEnv Specification

LNG-GeoEnv fully implements the OpenEnv standard:

✅ **Deterministic Environment**: Given seed and action sequence, output is reproducible.

✅ **Step/Reset API**: Full implementation of `reset()`, `step()`, `get_state()`.

✅ **Three Tasks**: Stable (easy), Volatile (medium), War (hard) with distinct configurations.

✅ **Continuous Reward Function**: Rewards provide signal at every step, not just episode end.

✅ **Agent Graders**: `EpisodeGrader` computes scores in [0.0, 1.0].

✅ **Reproducible Baseline**: `inference.py` runs and produces consistent scores across environments.

✅ **openenv.yaml**: Specifies tasks, entry point, runtime requirements.

✅ **Dockerized**: `Dockerfile` builds cleanly; `docker run` executes inference without errors.

✅ **Documentation**: README explains environment, tasks, action/observation spaces, setup.

---

## License

See [LICENSE](License) file.

