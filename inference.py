"""
inference.py — OpenEnv Hackathon Inference Script

Uses the OpenAI client with API_BASE_URL, MODEL_NAME, and HF_TOKEN env variables.
Must complete in under 20 minutes on vcpu=2, memory=8gb.

Usage:
    export API_BASE_URL=https://api.openai.com/v1
    export MODEL_NAME=gpt-4o-mini
    export HF_TOKEN=hf_...
    python inference.py
"""

import os
import json
from typing import Dict

from openai import OpenAI
from models import Action, ActionType, BaselineResult
from environment import DataCleaningEnv
from tasks import TASKS, run_grader


# ── Config from environment variables (required by hackathon spec) 

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     os.environ.get("OPENAI_API_KEY", ""))

TEMPERATURE  = 0.0
MAX_STEPS    = 15  # keeps runtime well under 20 min on 2vcpu/8gb



SYSTEM_PROMPT = """
You are a data cleaning agent. You will be given a summary of a dirty dataset
and must fix it step by step using JSON actions.

Each action must be a valid JSON object with this structure:
{
  "action_type": <one of the action types below>,
  "column": "<column name or null>",
  "fill_value": "<'mean' | 'median' | 'mode' | literal value | null>",
  "target_dtype": "<'int' | 'float' | 'str' | 'datetime' | null>",
  "normalize_map": {"raw": "canonical", ...} or null,
  "reference_table": "<table name or null>",
  "reference_column": "<column name or null>",
  "z_threshold": <float or null>
}

Available action types:
- fill_null        : Fill missing values in a column
- drop_duplicates  : Remove duplicate rows (no column needed)
- fix_dtype        : Cast a column to correct type
- normalize_str    : Standardize inconsistent string values
- drop_outliers    : Remove statistical outliers from a numeric column
- fix_foreign_key  : Nullify rows with invalid foreign key references
- fix_encoding     : Fix corrupted character encoding in a column
- done             : Signal that you are finished cleaning

Rules:
- Output ONLY a raw JSON object. No explanation, no markdown, no code fences.
- Choose the most impactful action at each step.
- When you think the data is clean enough, use action_type: "done".
"""


#Agent loop

def run_agent_on_task(client: OpenAI, task_id: str) -> float:
    """Run the agent on one task. Returns grader score 0.0–1.0."""

    env = DataCleaningEnv()
    obs = env.reset(task_id)
    initial_df   = env.original_df.copy()
    extra_tables = env._extra_tables.copy()

    history = []

    for step_num in range(MAX_STEPS):
        user_msg = f"""
Current dataset state (step {obs.step_number}):
- Task: {obs.task_id}
- Rows: {obs.total_rows} | Columns: {obs.total_columns}
- Nulls remaining: {obs.null_count}
- Duplicates remaining: {obs.duplicate_count}
- Quality score: {obs.quality_score:.3f}
- Issues: {', '.join(obs.issues) if obs.issues else 'None detected'}

Column details:
{_format_columns(obs)}

Choose your next cleaning action as a JSON object.
"""
        history.append({"role": "user", "content": user_msg})

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *history,
                ],
            )
            raw = response.choices[0].message.content.strip()
            history.append({"role": "assistant", "content": raw})
        except Exception as e:
            print(f"  [!] API error on step {step_num}: {e}")
            break

        try:
            action_dict = json.loads(raw)
            action = Action(**action_dict)
        except Exception as e:
            print(f"  [!] Could not parse action on step {step_num}: {e}")
            continue

        try:
            step_result = env.step(action)
            obs = step_result.observation
            print(f"  Step {step_num+1:2d} | {action.action_type:<20} | "
                  f"reward={step_result.reward.value:+.3f} | "
                  f"quality={obs.quality_score:.3f} | "
                  f"{step_result.reward.explanation}")
        except Exception as e:
            print(f"  [!] step() error: {e}")
            break

        if step_result.done:
            break

    result = run_grader(
        task_id=task_id,
        df_initial=initial_df,
        df_final=env.df,
        extra_tables=extra_tables,
    )
    return result.score


def _format_columns(obs) -> str:
    lines = []
    for col in obs.columns:
        lines.append(
            f"  - {col.name} (dtype={col.dtype}, nulls={col.null_count}, "
            f"unique={col.unique_count}, samples={col.sample_values[:3]})"
        )
    return "\n".join(lines)


#Main

def run_baseline() -> BaselineResult:
    api_key = HF_TOKEN or "dummy-key"
    base_url = API_BASE_URL or "https://api.openai.com/v1"

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        print(f"[!] Could not create OpenAI client: {e}")
        return BaselineResult(
            model=MODEL_NAME,
            results={"task_easy": 0.0, "task_medium": 0.0, "task_hard": 0.0},
            average=0.0
        )

    scores = {}

    for task_id, task in TASKS.items():
        print(f"\n{'='*60}")
        print(f"Task: {task['name']} ({task['difficulty'].upper()})")
        print(f"{'='*60}")
        try:
            score = run_agent_on_task(client, task_id)
        except Exception as e:
            print(f"[!] Task {task_id} failed: {e}")
            score = 0.0
        scores[task_id] = round(score, 4)
        print(f"\n  Final score: {score:.4f} | Passed: {score >= 0.7}")

    average = round(sum(scores.values()) / len(scores), 4)

    print(f"\n{'='*60}")
    print(f"BASELINE RESULTS")
    print(f"{'='*60}")
    for tid, s in scores.items():
        print(f"  {tid:<20} {s:.4f}")
    print(f"  {'Average':<20} {average:.4f}")

    return BaselineResult(model=MODEL_NAME, results=scores, average=average)

if __name__ == "__main__":
    try:
        run_baseline()
    except Exception as e:
        print(f"[!] Baseline failed: {e}")