# LNG-GeoEnv Vinay Setup

## Quick Start

### 1. Configure
```bash
cp .env.example .env
# Edit .env and use the injected hackathon proxy values:
# API_BASE_URL=https://...
# API_KEY=...
# MODEL_NAME=gpt-4.1-mini
```

### 2. Run
```bash
python inference.py
```

## Usage

| Command | Effect |
|---------|--------|
| `python main.py` | Run with LLM agent (recommended) |
| `python main.py --debug` | Test without LLM (baseline policy) |

## Configuration (.env)

```bash
API_BASE_URL=https://...            # REQUIRED: provided by validator
API_KEY=...                         # REQUIRED: provided by validator
MODEL_NAME=gpt-4.1-mini             # Optional override
```

## Core Features

- ✅ Gemini API integration with fallback
- ✅ Intelligent state formatting for LLM comprehension
- ✅ JSON response parsing with error handling
- ✅ Structured thinking prompts for reasoning
- ✅ Action guardrails and budget enforcement
- ✅ No repeated reroutes (route change protection)
- ✅ 2-step lookahead simulation for optimization
- ✅ Future-aware decision making

## Valid Actions

| Action | Parameters | Effect |
|--------|-----------|--------|
| wait | - | Do nothing |
| store | amount: 0-100 | Buy and store LNG |
| release | amount: 0-current | Release from storage |
| reroute | ship_id, new_route | Change ship's route |
| hedge | - | Buy options ($10) |

## Output Format

```json
{
  "episode_summary": {
    "total_reward": 5.54,
    "average_reward": 0.554,
    "steps_completed": 10,
    "final_storage": 130.0,
    "final_budget": 500.0
  },
  "metrics": {
    "total_cost": 132.25,
    "total_shortage": 23120.30,
    "total_risk": 1080.0,
    "shortage_events": 5
  },
  "actions": {
    "sequence": ["wait", "wait", ...],
    "distribution": {"wait": 10}
  },
  "agent_config": {
    "phase": 2,
    "model": "gemini-2.5-flash",
    "temperature": 0.7
  }
}
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| API vars not found | Add `API_BASE_URL` and `API_KEY` |
| Import error | `pip install openai` |
| Always "wait" action | Already working - baseline is conservative |
| Timeout | Use a smaller model via `MODEL_NAME` |

## Files

- `src/lng_geoenv/agent.py` - LLM agent (592 lines)
- `src/lng_geoenv/config.py` - Configuration management
- `.env.example` - Configuration template
- `main.py` - Entry point with episode runner
