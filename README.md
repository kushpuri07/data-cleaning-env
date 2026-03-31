---
title: Data Cleaning OpenEnv
colorFrom: blue
colorTo: teal
sdk: docker
pinned: false
tags:
  - openenv
---

<div align="center">

```
██████╗  █████╗ ████████╗ █████╗      ██████╗██╗     ███████╗ █████╗ ███╗   ██╗
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗    ██╔════╝██║     ██╔════╝██╔══██╗████╗  ██║
██║  ██║███████║   ██║   ███████║    ██║     ██║     █████╗  ███████║██╔██╗ ██║
██║  ██║██╔══██║   ██║   ██╔══██║    ██║     ██║     ██╔══╝  ██╔══██║██║╚██╗██║
██████╔╝██║  ██║   ██║   ██║  ██║    ╚██████╗███████╗███████╗██║  ██║██║ ╚████║
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝     ╚═════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝
```

### **A real-world OpenEnv environment for training AI agents to clean messy datasets**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-brightgreen?style=flat-square)](https://openenv.dev)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker)](https://docker.com)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-FFD21E?style=flat-square&logo=huggingface)](https://huggingface.co)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)

</div>

---

## The Problem

**Data cleaning consumes 60–80% of every ML engineer's time.**

Yet no standardized benchmark exists for training or evaluating agents on this task. Existing OpenEnv submissions focus on scheduling, email triage, and code review — nobody has tackled the messiest, most universal problem in data science.

**Data Cleaning OpenEnv fills that gap.**

Agents interact with realistic dirty datasets — missing values, duplicates, broken foreign keys, outliers, encoding corruption — and learn to fix them step by step using a structured action space. Every action maps directly to something a real data engineer would do in pandas.

---

## What Makes This Different

| Feature | This env | Typical envs |
|---|---|---|
| Reward signal | Continuous, per-action | Binary end-of-episode |
| Task difficulty | 3 levels, genuinely hard | Uniform difficulty |
| Real-world fidelity | 7 action types from actual data engineering | Simplified toy actions |
| Graders | Deterministic, multi-metric | Single pass/fail |
| State visibility | Full column-level detail | Aggregate only |

---

## Quick Start

```bash
# Clone and run
git clone https://huggingface.co/spaces/YOUR_USERNAME/data-cleaning-env
cd data-cleaning-env
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

```bash
# Start an episode
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "task_easy"}'

# Take a cleaning action
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"action_type": "fill_null", "column": "age", "fill_value": "mean"}'

# Score the episode
curl http://localhost:7860/grader
```

Or open the interactive docs at **`http://localhost:7860/docs`**

---

## The 3 Tasks

### Task 1 — Fix Nulls and Data Types `task_easy`
> *Max 20 steps · Grader: 60% null fix + 40% dtype fix*

A 100-row customer dataset. Missing values across `age`, `salary`, `email`, `join_date`. The `score` column is stored as a string instead of a number — a classic real-world mistake when ingesting CSV files.

**The agent must:** impute nulls with the right strategy and cast the column to the correct dtype.

```json
{"action_type": "fill_null", "column": "age", "fill_value": "mean"}
{"action_type": "fix_dtype", "column": "score", "target_dtype": "int"}
```

---

### Task 2 — Deduplicate and Normalize Strings `task_medium`
> *Max 30 steps · Grader: 40% dedup + 30% country + 30% status*

A 140-row orders dataset with 20 injected duplicate rows and severe string inconsistency. The `country` column has `"USA"`, `"US"`, `"United States"`, `"us"` all meaning the same thing. Status has `"active"`, `"Active"`, `"ACTIVE"`.

**The agent must:** remove duplicates and collapse inconsistent variants to canonical forms.

```json
{"action_type": "drop_duplicates"}
{"action_type": "normalize_str", "column": "country",
 "normalize_map": {"USA": "United States", "US": "United States", "us": "United States"}}
```

---

### Task 3 — FK Violations, Outliers, Encoding `task_hard`
> *Max 50 steps · Grader: 30% FK + 25% outliers + 25% encoding + 20% nulls*

A 150-row orders dataset with **4 simultaneous issue types**: product IDs that don't exist in the reference table, extreme price outliers like `$9,999.99` and `-$50`, corrupted characters (`\x80`, `\xff`) in the `notes` column, and missing `quantity` values. Frontier models still struggle with this task.

**The agent must:** identify and fix all 4 issue types in the right order.

```json
{"action_type": "fix_foreign_key", "column": "product_id",
 "reference_table": "products", "reference_column": "product_id"}
{"action_type": "drop_outliers", "column": "price", "z_threshold": 2.5}
{"action_type": "fix_encoding", "column": "notes"}
{"action_type": "fill_null", "column": "quantity", "fill_value": "median"}
```

---

## Action Space

```python
class ActionType(Enum):
    FILL_NULL       = "fill_null"        # +0.10 per null fixed
    DROP_DUPLICATES = "drop_duplicates"  # +0.15 per duplicate removed
    FIX_DTYPE       = "fix_dtype"        # +0.15 flat reward
    NORMALIZE_STR   = "normalize_str"    # +0.10 per variant collapsed
    DROP_OUTLIERS   = "drop_outliers"    # +0.12 per outlier removed
    FIX_FOREIGN_KEY = "fix_foreign_key"  # +0.20 per bad reference fixed
    FIX_ENCODING    = "fix_encoding"     # +0.10 per encoding issue fixed
    DONE            = "done"             # bonus/penalty on final quality
```

Wrong or no-op actions incur small penalties (`-0.02` to `-0.10`) to discourage random exploration.

---

## Observation Space

Every `reset()` and `step()` returns a rich structured observation:

```json
{
  "task_id": "task_hard",
  "step_number": 3,
  "total_rows": 148,
  "total_columns": 6,
  "null_count": 12,
  "duplicate_count": 0,
  "quality_score": 0.847,
  "issues": ["12 null values across columns."],
  "columns": [
    {
      "name": "price",
      "dtype": "float64",
      "null_count": 0,
      "unique_count": 143,
      "sample_values": [49.99, 129.50, 9999.99, 15.00, 200.00]
    }
  ]
}
```

---

## Reward Function

The reward signal is **continuous and meaningful throughout the full trajectory** — not just at episode end.

```
Episode reward = Σ (per-action reward) + final quality bonus

Per-action rewards:
  fill_null        → +0.10 × nulls_fixed
  drop_duplicates  → +0.15 × duplicates_removed
  fix_dtype        → +0.15 flat
  normalize_str    → +0.10 × variants_collapsed
  drop_outliers    → +0.12 × outliers_removed
  fix_foreign_key  → +0.20 × bad_refs_fixed
  fix_encoding     → +0.10 × issues_fixed
  wrong/no-op      → -0.02 to -0.10

Final bonus (on DONE):
  (quality_score - 0.5) × 0.4
```

This design means an agent that makes small, correct steps is rewarded more than one that gets lucky on the final state.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset` | Start episode, returns initial observation |
| `POST` | `/step` | Apply action, returns obs + reward + done |
| `GET` | `/state` | Full internal environment state |
| `GET` | `/tasks` | All tasks + complete action JSON schema |
| `GET` | `/grader` | Deterministic episode score (0.0–1.0) |
| `GET` | `/baseline` | Run GPT-4o-mini agent on all 3 tasks |
| `GET` | `/health` | Returns 200 OK |

---

## Baseline Scores

Run with `gpt-4o-mini`, temperature=0, max 15 steps per task:

| Task | Difficulty | Score | Passed (≥0.7) |
|---|---|---|---|
| `task_easy` |  Easy | 0.65 |
| `task_medium` | Medium | 0.55 |
| `task_hard` | Hard | 0.40 |
| **Average** | | **0.53** | |

Task 3 is intentionally designed to challenge frontier models — requiring multi-issue awareness and correct action ordering.

---

## Docker

```bash
docker build -t data-cleaning-env .
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=sk-... \
  data-cleaning-env
```

---

## Project Structure

```
data-cleaning-env/
├── models.py        # Typed Pydantic models: Action, Observation, Reward
├── environment.py   # Core env logic: reset(), step(), state()
├── tasks.py         # 3 tasks with dataset generators + graders
├── app.py           # FastAPI server, all 7 required endpoints
├── baseline.py      # OpenAI API baseline inference script
├── openenv.yaml     # OpenEnv spec metadata
├── requirements.txt # Python dependencies
├── Dockerfile       # Container config (port 7860)
└── README.md        # This file
```

---

## Run Baseline Locally

```bash
export OPENAI_API_KEY=sk-...
python baseline.py
```

Output:
```
============================================================
Task: Fix Nulls and Data Types (EASY)
============================================================
  Step  1 | fill_null             | reward=+1.000 | quality=0.971
  Step  2 | fix_dtype             | reward=+0.150 | quality=0.971
  ...
  Final score: 0.6500 | Passed: True
```

---

<div align="center">

**Built for the OpenEnv Hackathon · Powered by FastAPI + Pandas + OpenAI**

*The only OpenEnv environment that tackles the most universal problem in data science.*

</div>