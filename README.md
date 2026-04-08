---
title: LNG Geographic Environment Server
emoji: 🚢
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# LNG-GeoEnv: Real-World LNG Supply Chain Crisis Management

**Engineering decisions that keep the world running when supply chains break.**

> A reinforcement learning environment for training and evaluating agents on real-world liquefied natural gas (LNG) supply chain optimization under geopolitical disruptions.

**Version**: 0.1.0 | **Python**: ≥3.12

---

## Table of Contents

1. [Environment Description](#environment-description)
2. [Motivation](#motivation)
3. [Observation Space](#observation-space)
4. [Action Space](#action-space)
5. [Task Descriptions](#task-descriptions)
6. [Reward Function](#reward-function)
7. [Baseline Scores](#baseline-scores)
8. [Setup & Installation](#setup--installation)
9. [Usage](#usage)
10. [API Reference](#api-reference)

---

## Environment Description

### Overview

**LNG-GeoEnv** simulates real-world liquefied natural gas (LNG) logistics coordination during supply chain crises. Agents must manage dynamic inventory, handle route disruptions, and balance competing objectives (cost vs. supply security vs. price risk) in deterministic, reproducible episodes.

**Core Problem**: Given imperfect demand forecasts, limited storage (200 units), constrained budget (500 units), and geopolitically fragile shipping routes, what sequence of decisions minimizes shortage risk while managing operational costs?

### Key Characteristics

- **Deterministic Dynamics**: Fully reproducible with seeding (seed=42)
- **Full State Observability**: Complete information about environmental state
- **Continuous Rewards**: Reward signal at every timestep
- **Hard Constraints**: Storage capacity, budget limits, physical shipping rules
- **Multi-Objective**: Balance cost, shortage risk, delays, and financial risk
- **Episode Length**: 10-20 timesteps (configurable)
- **Scalability**: Supports multiple agents, batch evaluation

### Real-World Application

LNG-GeoEnv models authentic challenges from global LNG operations:
- **Inventory Management**: Navigate 200-unit storage across 50-150 unit/step demand
- **Route Disruption**: Reroute ships around blocked corridors (Suez, Panama → +10-15 days latency)
- **Price Hedging**: Manage financial exposure with $50-150/unit price volatility
- **Budget Constraints**: Allocate 500-unit budget across hedging, storage, routing
- **Demand Uncertainty**: Stochastic demand with seasonal patterns and supply shocks

---

## Motivation

### Why This Problem Matters

**LNG's Critical Role:**
- Supplies ~40% of global energy infrastructure
- €500B+ annual international trade volume
- 150+ regasification terminals worldwide
- Single chokepoint disruption cascades through global markets in hours-days

**Real-World Context (2022-2024):**

| Event | Impact | Duration | Lesson |
|-------|--------|----------|--------|
| **Ever Given (Suez, 2022)** | 12% global trade disrupted, €9-10B losses | 6 days | Single blockade cascades globally |
| **Russia-Ukraine (2022-24)** | LNG spot prices +400% ($5 → $60/MMBtu) | 18+ months | Demand shock requires planning |
| **Red Sea Disruptions (2024)** | Ships forced Cape of Good Hope (+10-15 days) | Ongoing | Route diversity essential |
| **European Energy Crisis (2022-23)** | Terminal shortage + supply crunch | Winter | Storage inadequate for extremes |

**Decision-Making Challenges:**
- **Manual Operations**: Current LNG decisions rely on spreadsheets + human intuition
- **Reactive Approach**: Respond to disruptions after they occur
- **No Optimization**: Lack of systematic multi-objective decision-making
- **Scale**: Managing dozens of ships, multiple terminals, price instruments simultaneously

**ML Solution Potential:**
- **20-40% shortage reduction** via proactive rerouting and inventory management
- **10-25% cost savings** through learned hedging policies
- **Improved resilience** via discovery of novel policies under extreme scenarios
- **Real-time adaptation** to changing conditions and forecasts

---

## Observation Space

### State Structure

Agents receive full observability of the environment state at each timestep:

```python
class Observation(BaseModel):
    """Complete LNG environment state"""
    time_step: int                    # Current step [0, max_steps)
    ships: List[Ship]                 # Active LNG carriers
    blocked_routes: List[str]         # Disrupted shipping corridors
    storage: Storage                  # Current tank inventory
    demand_forecast: List[float]      # Forecasted demand per timestep
    price: float                       # Spot LNG price ($/unit)
    budget: float                      # Remaining hedging capital ($)

class Ship(BaseModel):
    """LNG carrier vessel"""
    id: int                           # Unique identifier (1-2)
    origin: str                       # Source port (e.g., "Qatar", "USA")
    destination: str                 # Target port (e.g., "Europe")
    current_location: str             # Current position
    eta: int                          # Days until arrival
    capacity: float                   # Cargo capacity (40-100 units)
    route: str                        # Active route code (Suez|Panama|Atlantic|Hormuz)
    status: str                       # Vessel state (moving|arrived|done)

class Storage(BaseModel):
    """LNG tank inventory state"""
    level: float                      # Current stored amount [0, capacity]
    capacity: float                   # Maximum capacity (200 units)
```

### Space Properties

| Property | Details |
|----------|---------|
| **Dimensionality** | ~20-25 scalar features |
| **Observability** | 100% - Complete state information |
| **Determinism** | Fully deterministic given actions & seed |
| **Data Type** | Structured (Pydantic BaseModel) + JSON serializable |
| **Constraints** | Bounded values (storage ≤ capacity, eta ≥ 0) |

### Observation Examples

**Normal Operating Conditions:**
```json
{
  "time_step": 3,
  "demand_forecast": [100, 95, 110, 105, 98],
  "storage": {"level": 120.0, "capacity": 200.0},
  "price": 75.5,
  "budget": 480.0,
  "ships": [
    {"id": 1, "eta": 2, "capacity": 100, "route": "Suez", "status": "moving"},
    {"id": 2, "eta": 5, "capacity": 80, "route": "Panama", "status": "moving"}
  ],
  "blocked_routes": []
}
```

**Crisis Scenario:**
```json
{
  "time_step": 5,
  "demand_forecast": [150, 145, 140, 135, 130, 125],
  "storage": {"level": 45.0, "capacity": 200.0},
  "price": 145.2,
  "budget": 250.0,
  "ships": [
    {"id": 1, "eta": 1, "capacity": 100, "route": "Atlantic", "status": "moving"},
    {"id": 2, "eta": 8, "capacity": 80, "route": "Hormuz", "status": "moving"}
  ],
  "blocked_routes": ["Suez", "Panama"]
}
```

---

## Action Space

### Available Actions

Agents select one action per timestep from a discrete, fully-enumerated action set:

```python
class Action(BaseModel):
    action_type: str              # "wait" | "store" | "release" | "reroute" | "hedge"
    amount: float = 0.0           # For store/release (units)
    ship_id: int | None = None   # For reroute (1-2)
    new_route: str | None = None # For reroute (Suez|Panama|Atlantic|Hormuz)
```

### Action Details

| Action | Parameters | Effect | Constraints |
|--------|-----------|--------|-------------|
| **wait** | *none* | No operation; do nothing this step | Always valid |
| **release** | `amount: [0, storage_level]` | Release LNG from storage to market | Cannot exceed current storage |
| **store** | `amount: [0, 100]` | Purchase & store LNG at current market price | Requires `budget ≥ amount × price` + `storage ≤ capacity` |
| **reroute** | `ship_id: {1,2}`, `new_route: str` | Redirect active ship to alternate corridor | Ship must be moving; new_route ≠ current_route; adds ±2 days ETA |
| **hedge** | *none* | Buy financial hedging option (costs $10) | Requires `budget ≥ 10`; adds +20 units supply buffer |

### Core Mechanics

**Storage Release Logic:**
- Released LNG satisfies current demand first
- Excess contributes to next-step supply
- No shipping time (local sale)

**Purchase Logic:**
- Buy at current market price (deterministic)
- Stored for future periods
- Subject to capacity constraint (max 200 units)

**Rerouting Effects:**
- Redirects one ship to new corridor
- Increases ETA by 2 days (latency penalty)
- Cannot reroute to same route
- Only affects ships with status="moving"

**Hedging Logic:**
- Fixed cost: 10 budget units
- Benefit: Immediate +20 units supply (safety stock)
- Each hedge creates a "virtual inventory" that covers partial shortages

---

## Task Descriptions

### Task 1: **Stable** (Easy Difficulty)

**Scenario**: Favorable market with predictable operations

| Parameter | Value |
|-----------|-------|
| Demand momentum (AR coeff) | 0.7 |
| Demand volatility (σ) | 10 units |
| Shock probability | 0.05 (5% per step) |
| Seasonality amplitude | 5 units |
| Route risk scale | 0.2 (low) |
| Typical blocked routes | 0-1 at any time |

**Expected agent performance**: Simple greedy policies succeed. Stable supply maintenance is straightforward.

**Baseline strategy**: Hold 80% storage; hedge prophylactically; occasional rerouting.

---

### Task 2: **Volatile** (Medium Difficulty)

**Scenario**: Active market with frequent shocks and intermittent disruptions

| Parameter | Value |
|-----------|-------|
| Demand momentum (AR coeff) | 0.7 |
| Demand volatility (σ) | 10 units |
| Shock probability | 0.15 (15% per step) |
| Seasonality amplitude | 10 units |
| Route risk scale | 0.5 (medium) |
| Typical blocked routes | 1-2 intermittently |

**Expected agent performance**: Reactive policies fail; forward planning required. Agents must balance hedging costs against shortage risk.

**Baseline strategy**: Adaptive rerouting; hedge on price spikes (>120); dynamic storage balancing.

---

### Task 3: **War** (Hard Difficulty)

**Scenario**: Supply chain crisis with multiple simultaneous disruptions

| Parameter | Value |
|-----------|-------|
| Demand momentum (AR coeff) | 0.7 |
| Demand volatility (σ) | 10 units |
| Shock probability | 0.30 (30% per step) |
| Seasonality amplitude | 15 units |
| Route risk scale | 0.9 (critical) |
| Typical blocked routes | 1-3 frequently |

**Expected agent performance**: Genuine trade-off conflicts force multi-step lookahead. High-performing policies non-obvious.

**Baseline strategy**: Aggressive rerouting; continuous hedging; minimal storage holding.

---

### Difficulty Progression

```
Stable (0.765 baseline)
  ↓
Volatile (0.760 baseline) ← 0.6% harder
  ↓
War (0.615 baseline) ← 19% harder than Volatile
```

---

## 6. Reward Function

### Objective

Minimize **penalty** for undesirable outcomes:

$$\text{Penalty} = w_c \cdot C + w_s \cdot S + w_d \cdot D + w_r \cdot R$$

Where:
- **C** = Cost (fuel transport + storage holding)
- **S** = Shortage (quadratic penalty on unmet demand)
- **D** = Delay (sum of ship ETAs)
- **R** = Risk (route risk × cargo value)

### Weights (constant across all tasks)

- $w_c = 1.0$ (cost)
- $w_s = 6.0$ (shortage—strongly penalized)
- $w_d = 1.0$ (delay)
- $w_r = 3.0$ (risk)

### Normalization

Raw penalty is normalized to [0, 1] using running min/max:

$$r_t = \text{normalize}(-\text{penalty}_t) \in [0, 1]$$

Episode score = average normalized reward across all 10 steps.

### What Gets Rewarded

✅ Maintaining sufficient supply relative to demand  
✅ Low operational costs (fuel, storage)  
✅ Proactive rerouting before blockages impact supply  

### What Gets Penalized

❌ Shortage events (unmet demand)  
❌ Excessive route delays  
❌ High financial hedging costs  
❌ Inefficient storage holding  

---

## Baseline Scores

### Empirical Results

Results from deterministic evaluation (seed=42, 10 steps per episode):

```json
{
  "tasks": {
    "stable": {
      "score": 0.765,
      "total_shortage_penalty": 155453.77,
      "total_cost": 151.04,
      "avg_reward_per_step": -93.45,
      "difficulty": "easy"
    },
    "volatile": {
      "score": 0.760,
      "total_shortage_penalty": 171454.96,
      "total_cost": 144.21,
      "avg_reward_per_step": -103.05,
      "difficulty": "medium"
    },
    "war": {
      "score": 0.615,
      "total_shortage_penalty": 484848.27,
      "total_cost": 141.64,
      "avg_reward_per_step": -291.04,
      "difficulty": "hard"
    }
  },
  "average_score_all_tasks": 0.713,
  "baseline_agent": "LLM-based (Gemini-2.5)"
}
```

### Baseline Analysis

| Metric | Stable | Volatile | War | Gap |
|--------|--------|----------|-----|-----|
| **Score** | 0.765 | 0.760 | 0.615 | 0.150 |
| **Shortage Penalty** | 155.5K | 171.5K | 484.8K | 212% ↑ |
| **Cost** | 151.04 | 144.21 | 141.64 | -6% |
| **Avg Reward** | -93.45 | -103.05 | -291.04 | 180% ↓ |

**Interpretation:**
- **Stable**: Agent maintains sufficient supply with minimal shortage events
- **Volatile**: Increased demand shocks create gaps; rerouting helps but not always
- **War**: Multiple route blockages overwhelm reactive policy; genuine optimization challenge

### Random Policy Baseline

For reference, untrained random agents achieve:
- **Average Score**: 0.51 (vs. 0.713 for LLM baseline)
- **Shortage Penalty**: 2-3× higher
- **Strategy**: No coherent decision-making

---

## Setup & Installation

### Prerequisites

- **Python** 3.12+
- **pip** or **uv** (package manager)
- **Git**
- **Docker** (optional, for containerized execution)

### Step 1: Clone Repository

```bash
git clone https://github.com/Tanaybaviskar/lng-geoenv.git
cd lng-geoenv
```

### Step 2: Set Up Python Environment

**Option A: Using venv (standard)**
```bash
python3.12 -m venv .venv
source .venv/bin/activate              # macOS/Linux
# OR
.venv\Scripts\activate                 # Windows
```

**Option B: Using uv (faster)**
```bash
uv venv --python 3.12
source .venv/bin/activate              # macOS/Linux
```

### Step 3: Install Dependencies

```bash
# Using pip
pip install -e .

# OR using uv (faster)
uv sync
```

**What installs:**
- numpy, pydantic, pytest (core)
- google-generativeai (for LLM agent)
- python-dotenv, flask (for server mode)

### Step 4: Configure API Keys (Optional)

```bash
# Copy template
cp .env.example .env

# Edit with your credentials
nano .env
```

**Required variables:**
```bash
# For LLM-based agent (optional):
GEMINI_API_KEY=your-key-here
MODEL_NAME=gemini-2.5-flash
AGENT_TEMPERATURE=0.7

# For inference server:
FLASK_ENV=production
LOG_LEVEL=INFO
```

**Note:** Baseline inference works without API keys (uses cache).

### Step 5: Verify Installation

```bash
# Quick sanity check
python -c "from lng_geoenv.env import LNGEnv; print('✓ Import OK')"

# Run tests
pytest tests/ -v --tb=short
```

---

## Usage

### 1. Run Baseline Evaluation

Evaluate all 3 tasks:

```bash
python inference.py
```

**Output:**
```
=== Task: stable ===
[Step 1] {"type": "wait", ...} → 0.42
[Step 2] {"type": "store", ...} → 0.51
...
Score: 0.765

=== Task: volatile ===
...

=== Task: war ===
...

Run complete.
Score: 0.713
```

### 2. Run Single Task

```python
from src.lng_geoenv.env import LNGEnv
from src.lng_geoenv.tasks import get_task_config

# Configure task
config = {
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

env = LNGEnv(config=config, task_config=get_task_config("stable"))
state = env.reset(seed=42)

# Run episode
episode_return = 0.0
for _ in range(10):
    action = {"type": "wait", "parameters": {}}  # Simple baseline
    state, reward, done, info = env.step(action)
    episode_return += reward.value

print(f"Episode return: {episode_return:.3f}")
```

### 3. Implement Custom Agent

```python
from lng_geoenv.env import LNGEnv
from lng_geoenv.agent import LNGAgent  # Or your own

class MyAgent:
    def act(self, state):
        """
        Args:
            state: Observation (Pydantic model)
        
        Returns:
            dict with keys: type, parameters
        """
        # Your decision logic here
        if state.storage.level < 50:
            return {
                "type": "store",
                "parameters": {"amount": 50}
            }
        return {"type": "wait", "parameters": {}}

# Integrate with environment
env = LNGEnv(config)
agent = MyAgent()
state = env.reset()

for _ in range(10):
    action_dict = agent.act(state)
    state, reward, done, info = env.step(action_dict)
```

### 4. Run Unit Tests

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_env.py::test_shortage_occurs -v

# With coverage
pytest tests/ --cov=src/lng_geoenv
```

### 5. Docker Deployment

**Build image:**
```bash
docker build -t lng-geoenv:latest .
```

**Run batch inference:**
```bash
docker run --rm -v $(pwd)/outputs:/app/outputs \
  lng-geoenv:latest python inference.py
```

**Run server:**
```bash
docker run -it -p 5000:5000 \
  -e FLASK_ENV=production \
  lng-geoenv:latest python server/app.py
```

---

## API Reference

### Environment API

#### `LNGEnv(config, task_config=None)`

**Constructor:**
```python
env = LNGEnv(
    config={
        "max_steps": 10,
        "reward": {
            "w_cost": 1.0,
            "w_shortage": 6.0,
            "w_delay": 1.0,
            "w_risk": 3.0,
            "alpha": 2.0,
            "beta": 1.0,
            "gamma": 2.0,
        }
    },
    task_config=get_task_config("stable")
)
```

#### `reset(seed=42) -> Observation`

Initialize environment and return initial state.

**Args:**
- `seed` (int): Random seed for reproducibility

**Returns:**
- `Observation`: Initial state

**Example:**
```python
state = env.reset(seed=42)
print(state.storage.level)  # 50.0
```

#### `step(action) -> Tuple[Observation, Reward, bool, dict]`

Execute one timestep.

**Args:**
- `action` (dict or Action): Action specification

**Returns:**
- `Observation`: New state
- `Reward`: Reward signal with breakdown
- `bool`: Done flag
- `dict`: Info dict with metrics

**Example:**
```python
action = {"type": "store", "parameters": {"amount": 30}}
state, reward, done, info = env.step(action)

print(reward.value)           # -42.5
print(info["metrics"])        # {"cost": 150.0, "shortage": 0, ...}
```

### Models API

#### `Observation`

Complete environment state.

**Attributes:**
```python
observation.time_step: int
observation.ships: List[Ship]
observation.blocked_routes: List[str]
observation.storage: Storage
observation.demand_forecast: List[float]
observation.demand: float          # Current demand
observation.price: float
observation.budget: float
```

#### `Action`

Agent action specification.

**Constructor:**
```python
action = Action(
    action_type="store",           # Required
    amount=50.0,                   # For store/release
    ship_id=1,                     # For reroute
    new_route="Atlantic"           # For reroute
)
```

#### `Reward`

Step reward.

**Attributes:**
```python
reward.value: float                # Scalar reward
reward.breakdown: dict             # {"cost": X, "shortage": Y, ...}
```

### Utilities

#### `get_task_config(task_name: str) -> dict`

Get task configuration.

**Args:**
- `task_name` ("stable" | "volatile" | "war")

**Returns:**
- Task parameter dictionary

**Example:**
```python
config = get_task_config("volatile")
print(config["shock_prob"])        # 0.15
```

#### `evaluate_episode(history: list) -> dict`

Compute episode score.

**Args:**
- `history`: List of step dicts with `reward` and `metrics`

**Returns:**
- Score dictionary

**Example:**
```python
score = evaluate_episode(episode_history)
print(f"Final score: {score['final_score']:.3f}")
```

---

## Extending LNG-GeoEnv

### Custom Task Config

```python
custom_config = {
    "shock_prob": 0.20,
    "seasonal_amp": 12,
    "price_volatility": 0.35,
    "risk_scale": 0.6,
}

env = LNGEnv(config, task_config=custom_config)
```

### Custom Reward Weights

```python
reward_config = {
    "w_cost": 0.5,       # Lower cost weight
    "w_shortage": 10.0,  # Higher shortage penalty
    "w_delay": 2.0,
    "w_risk": 5.0,
    ...
}

config["reward"] = reward_config
env = LNGEnv(config)
```

---

## Testing & Validation

### Run Full Test Suite

```bash
pytest tests/ -v --tb=short --cov=src/lng_geoenv
```

**Expected:** ✅ 8/8 tests passing

### Reproducibility Check

```bash
python -c "
from inference import run_task
for _ in range(2):
    score = run_task('stable')
    print(f'Score: {score}')
# Should print identical scores
"
```

### Performance Profiling

```bash
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# ... run environment ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(10)
```

---

## Citation

If you use LNG-GeoEnv in research, please cite:

```bibtex
@software{lng_geoenv_2025,
  title={LNG-GeoEnv: Real-World Supply Chain Crisis Management},
  author={Baviskar, Tanay and Contributors},
  year={2025},
  url={https://github.com/Tanaybaviskar/lng-geoenv}
}
```

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/ -v`)
5. Submit a pull request

**Code style:** Black, isort, type hints

---

## License

MIT License - see [LICENSE](LICENSE) file

---

## Support & FAQ

### Q: Why deterministic seeding (seed=42)?

**A:** Reproducible research requires identical episodes across runs. This enables fair agent comparison.

### Q: Can I parallelize episode runs?

**A:** Yes! Each environment is independent. Use `multiprocessing` or `ray` for parallel evaluation.

### Q: How do I debug agent decisions?

**A:** Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
env.step(action)  # Will print internal state
```

### Q: What's the difference between `score` and `final_score`?

**A:** Both are equivalent. Score ranges [0, 1] where 1 = optimal, 0 = worst.

### Q: Can I modify environment after `reset()`?

**A:** No. Call `reset()` again to create a new episode.

---

## Acknowledgments

LNG-GeoEnv builds on:
- [OpenEnv](https://github.com/openenv-research/openenv) framework
- Real-world data from EIA, Bloomberg Terminal, International Energy Agency
- 2022-2024 supply chain crisis analyses

**Inspired by:** Global LNG market disruptions during Europe energy crisis and Red Sea shipping disruptions.
