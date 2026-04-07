import os
import json
from openai import OpenAI
from models import Action, BaselineResult
from environment import DataCleaningEnv
from tasks import TASKS, run_grader

API_BASE_URL = os.getenv("API_BASE_URL", "<your-api-base-url>")
MODEL_NAME   = os.getenv("MODEL_NAME",   "<your-active-model>")
HF_TOKEN     = os.getenv("HF_TOKEN")

SYSTEM_PROMPT = """You are a data cleaning agent. Output ONLY a raw JSON action object.
Available action_types: fill_null, drop_duplicates, fix_dtype, normalize_str, drop_outliers, fix_foreign_key, fix_encoding, done
JSON format: {"action_type": "...", "column": "...", "fill_value": "...", "target_dtype": "...", "normalize_map": {}, "reference_table": "...", "reference_column": "...", "z_threshold": 3.0}"""

def run_baseline():
    client = OpenAI(api_key=HF_TOKEN or "dummy-key", base_url=API_BASE_URL)

    for task_id, task in TASKS.items():
        print(f"START task={task_id}")

        env = DataCleaningEnv()
        obs = env.reset(task_id)
        initial_df   = env.original_df.copy()
        extra_tables = env._extra_tables.copy()
        history = []

        for step_num in range(15):
            user_msg = f"""Dataset state:
- Rows: {obs.total_rows}, Nulls: {obs.null_count}, Duplicates: {obs.duplicate_count}, Quality: {obs.quality_score:.3f}
- Issues: {', '.join(obs.issues) if obs.issues else 'None'}
- Columns: {[f"{c.name}(nulls={c.null_count},dtype={c.dtype})" for c in obs.columns]}
Output a JSON action."""

            history.append({"role": "user", "content": user_msg})

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *history,
                    ],
                )
                raw = response.choices[0].message.content.strip()
                history.append({"role": "assistant", "content": raw})
            except Exception as e:
                print(f"STEP task={task_id} step={step_num} action=none reward=0.0 error={e}")
                break

            try:
                action = Action(**json.loads(raw))
                result = env.step(action)
                obs    = result.observation
                print(f"STEP task={task_id} step={step_num} action={action.action_type} reward={result.reward.value:.4f} quality={obs.quality_score:.4f}")
            except Exception as e:
                print(f"STEP task={task_id} step={step_num} action=parse_error reward=0.0 error={e}")
                continue

            if result.done:
                break

        result = run_grader(task_id, initial_df, env.df, extra_tables)
        print(f"END task={task_id} score={result.score:.4f} passed={result.passed}")

if __name__ == "__main__":
    try:
        run_baseline()
    except Exception as e:
        print(f"END error={e}")