# LNG-GeoEnv: Real-World LNG Supply Chain Crisis Management

**Engineering decisions that keep the world running when supply chains break.**

> A reinforcement learning environment for training and evaluating agents on real-world liquefied natural gas (LNG) supply chain optimization under geopolitical disruptions.

---

## 1. Environment Description

### What is LNG-GeoEnv?

LNG-GeoEnv simulates **real-world LNG logistics coordination** during supply chain crises. Agents must manage dynamic inventory, handle route disruptions, and balance competing objectives (cost vs. supply security vs. price risk) in deterministic, reproducible episodes.

**The core problem**: Given imperfect demand forecasts, limited storage, constrained budget, and geopolitically fragile shipping routes, what sequence of decisions minimizes shortage risk while managing operational costs?

**Real tasks agents face in LNG operations:**
- Allocate limited storage (200 units) across fluctuating demand (50-150 units/step)
- Reroute ships around blocked corridors (Suez, Panama blocked → add 10-15 days latency)
- Hedge financial price exposure ($50-150/unit volatility)
- Manage fixed budget (500 units) across hedging, storage, and routing operations

### Key Features

✅ **Full State Observability**: Agents see ships, storage, demand, price, blockages  
✅ **Continuous Reward Signal**: Rewards at every step (not just episode end)  
✅ **Hard Constraints**: Storage capacity, budget limits, ship routing rules  
✅ **Deterministic Seeding**: 100% reproducible (seed=42)  
✅ **Three Difficulty Levels**: Easy → Medium → Hard task progression  
✅ **Real-World Mechanics**: Based on EIA, Bloomberg LNG data and 2022-2024 crises  

---

## 2. Real-World Motivation

### Why This Problem Matters

**LNG is critical global infrastructure:**
- Supplies 40% of global energy
- €500B+ annual trade volume
- Single route blockade cascades through global markets in days

**Recent real-world examples:**
- **Ever Given (2022)**: 6-day Suez blockade disrupted 12% of global trade, triggered energy crisis
- **Red Sea (2024)**: Houthi attacks forced Cape of Good Hope rerouting (+10-15 days per voyage)
- **Europe 2022-23**: LNG terminal shortage → spot prices hit $60/MMBtu (10× baseline)

**The decision-making gap:**
Current LNG operations rely on:
- Manual spreadsheets + human intuition
- Slow, reactive decision-making
- No systematic optimization under uncertainty

**What agents could improve:**
- Reduce shortage incidents by 20-40% via proactive rerouting
- Lower hedging costs via learned price sensitivity
- Discover novel inventory policies under extreme disruption

---

## 3. Observation Space

Agents observe the complete state at each timestep as a Pydantic model:

```python
class Observation(BaseModel):
    time_step: int                    # Current step [0, 9]
    ships: list[Ship]                 # Active vessels (ship objects)
    blocked_routes: list[str]         # Currently disrupted corridors
    storage: Storage                  # Level & capacity
    demand: float                      # Current timestep demand (units)
    price: float                       # Spot LNG price ($/unit)
    budget: float                      # Remaining hedging capital
    
class Ship(TypedDict):
    id: int                           # 1 or 2
    origin: str                       # e.g., "Qatar"
    destination: str                 # e.g., "Europe"
    eta: int                          # Days until arrival
    capacity: float                   # Cargo size (units)
    route: str                        # Current route (Suez|Panama|Atlantic|Hormuz)
    status: str                       # "moving"|"arrived"|"done"
    
class Storage(TypedDict):
    level: float                      # Current level (0-200 units)
    capacity: float                   # Max 200 units
```

**Properties:**
- Full observability: No hidden information
- Deterministic given seed & action history
- ~15-20 scalar features total

---

## 4. Action Space

Agents take one action per timestep. Valid action types:

| Action | Parameters | Effect | Cost |
|--------|-----------|--------|------|
| **wait** | `{}` | No operation | 0 |
| **release** | `{"amount": float}` | Release stored LNG to market | 0 |
| **store** | `{"amount": float}` | Buy & store LNG (if budget allows) | amount × price |
| **reroute** | `{"ship_id": int, "new_route": str}` | Redirect ship to different corridor | +2 days ETA |
| **hedge** | `{}` | Financial protection against price spikes | 10 budget units |

**Constraints:**
- release: Cannot exceed current storage
- store: Blocked by insufficient budget or capacity
- reroute: New route ≠ current route, ship must be "moving"
- hedge: Fixed cost of 10; adds +20 units supply as safety margin

---

## 5. Task Descriptions

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

## 7. Baseline Scores

Results from `python inference.py` (deterministic, seed=42):

```json
{
  "environment": "lng-geoenv",
  "tasks": [
    {
      "task": "stable",
      "score": 0.765,
      "risk_adjusted_score": 0.765,
      "breakdown": {
        "cost": 151.04,
        "shortage": 155453.77
      },
      "total_reward": -934.53,
      "avg_reward": -93.45,
      "steps": 10
    },
    {
      "task": "volatile",
      "score": 0.760,
      "risk_adjusted_score": 0.760,
      "breakdown": {
        "cost": 144.21,
        "shortage": 171454.96
      },
      "total_reward": -1030.53,
      "avg_reward": -103.05,
      "steps": 10
    },
    {
      "task": "war",
      "score": 0.615,
      "risk_adjusted_score": 0.615,
      "breakdown": {
        "cost": 141.64,
        "shortage": 484848.27
      },
      "total_reward": -2910.35,
      "avg_reward": -291.04,
      "steps": 10
    }
  ],
  "average_score": 0.713,
  "execution_status": "success"
}
```

**Key Properties:**
✅ Scores are **non-zero** (0.615-0.765)  
✅ Scores in **[0, 1]** range  
✅ Ordering: **Stable > Volatile > War** ✓  
✅ **Deterministic** (identical across runs)  

---

## 8. Setup & Installation

### Prerequisites

- Python 3.12+
- Git
- Docker (optional, for containerized deployment)

### Local Setup

```bash
# Clone repository
git clone https://github.com/Tanaybaviskar/lng-geoenv.git
cd lng-geoenv

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
# OR using uv (faster):
uv sync

# Set up environment file
cp .env.example .env
# Edit .env with your API key (optional for baseline)
nano .env
```

### Run Baseline Inference

```bash
# Test reproducibility (run twice, should be identical)
python inference.py > run1.json
python inference.py > run2.json
diff run1.json run2.json  # Should be empty
```

**Output:**
```json
{
  "environment": "lng-geoenv",
  "tasks": [...],
  "average_score": 0.713,
  "execution_status": "success"
}
```

### Run Tests

```bash
pytest tests/ -v

# Expected: 8/8 tests passing
```

### Validate OpenEnv Compliance

```bash
openenv validate

# Expected:
# [OK] lng-geoenv: Ready for multi-mode deployment
```

---

## 9. Docker Deployment

### Build Image

```bash
docker build -t lng-geoenv .
```

### Run Inference (Batch Mode)

```bash
docker run --env-file .env lng-geoenv
```

Outputs JSON to stdout.

### Run Server (API Mode)

```bash
docker run -p 8000:8000 --env-file .env lng-geoenv python server/app.py
```

Then test endpoints:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/inference
curl http://localhost:8000/task/stable
```

---

## 10. API Endpoints (Server Mode)

When `server/app.py` is running:

### `GET /health`
Health check.
```bash
curl http://localhost:8000/health
```

### `GET /inference`
Run all 3 tasks.
```bash
curl http://localhost:8000/inference
```

### `GET /task/<task_name>`
Run single task (stable|volatile|war).
```bash
curl http://localhost:8000/task/stable
```

### `GET /`
API documentation.

---

## 11. Hugging Face Spaces Deployment

### Step 1: Create Space

Visit [huggingface.co/spaces](https://huggingface.co/spaces):
1. **Create new Space**
2. **Name**: `lng-geoenv`
3. **SDK**: Docker
4. **Visibility**: Public
5. **Connect GitHub**: `Tanaybaviskar/lng-geoenv`

### Step 2: Automatic Deployment

Hugging Face will:
- Pull from GitHub
- Build Dockerfile
- Deploy on port 7860
- Expose at `https://huggingface.co/spaces/{username}/lng-geoenv`

### Step 3: Access Deployed Environment

```bash
# Health check
curl https://huggingface.co/spaces/{username}/lng-geoenv/health

# Run inference
curl https://huggingface.co/spaces/{username}/lng-geoenv/inference
```

---

## 12. Project Structure

```
lng-geoenv/
├── src/lng_geoenv/
│   ├── env.py                       # LNGEnv core (step, reset, get_state)
│   ├── models.py                    # Pydantic types (Observation, Action, Reward)
│   ├── runner.py                    # run_task(task_name) orchestrator
│   ├── evaluator.py                 # evaluate_episode(history) → score
│   ├── reward.py                    # Reward computation engine
│   ├── grader.py                    # Normalization logic
│   ├── demand.py                    # DemandGenerator (AR(1) + shocks)
│   ├── world.py                     # Ship dynamics & route risk
│   ├── agent.py                     # Gemini LLM agent (optional)
│   └── tasks.py                     # Task configurations
├── server/
│   └── app.py                       # Flask API server
├── tests/                           # Test Suite
│   ├── test_env.py
│   ├── test_reward.py
│   ├── test_env_integration.py
│   ├── test_env.py
│   ├── test_models.py
│   ├── test_world_tasks_evaluator.py
│   └── test_demand.py
├── inference.py                     # Baseline entry point
├── openenv.yaml                     # OpenEnv spec
├── Dockerfile
├── pyproject.toml
├── README.md                        # Documentation
└── LICENSE
```

---

## 13. Testing & Validation

### Full Validation

```bash
# 1. Local tests
openenv validate
pytest tests/ -v

# 2. Reproducibility check
python inference.py > /tmp/r1.json
python inference.py > /tmp/r2.json
diff /tmp/r1.json /tmp/r2.json

# 3. Docker build & run
docker build -t lng-geoenv .
docker run --env-file .env lng-geoenv

# 4. Server test
docker run -p 8000:8000 --env-file .env lng-geoenv python server/app.py &
sleep 2
curl http://localhost:8000/inference | jq '.average_score'
```

### Expected Results

| Check | Result |
|-------|--------|
| `openenv validate` | OK ✓ |
| Test suite | 8/8 passing ✓ |
| Reproducibility | Identical JSON ✓ |
| Docker build | Success ✓ |
| Docker inference | Valid JSON ✓ |
| Task ordering | stable > volatile > war ✓ |
| Score range | [0.615, 0.765] ✓ |

---

## 14. Citation

If you use LNG-GeoEnv in research, please cite:

```bibtex
@software{lnggeoenv2024,
  title={LNG-GeoEnv: Real-World Supply Chain Crisis Management for RL},
  author={Baviskar, Tanay and others},
  year={2024},
  url={https://github.com/Tanaybaviskar/lng-geoenv}
}
```

---

## 15. License

MIT License — see [LICENSE](LICENSE) file.

---

## 16. Support & Contribution

**Questions?** Open a GitHub issue.  
**Pull requests** welcome!  
**Community**: Experiments, benchmarks, novel agents all encouraged.

---

**Last Updated**: March 2024  
**Status**: Production-ready ✓  
**OpenEnv Compliance**: Validated ✓
