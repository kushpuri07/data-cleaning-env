import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Environment variable configuration per guidelines
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Import heavy dependencies after env vars are set
try:
    from openai import OpenAI
    from models import Action, BaselineResult
    from environment import DataCleaningEnv
    from tasks import TASKS, run_grader
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)


SYSTEM_PROMPT = """You are a data cleaning agent. Output ONLY a raw JSON action object.
Available action_types: fill_null, drop_duplicates, fix_dtype, normalize_str, drop_outliers, fix_foreign_key, fix_encoding, done
JSON format: {"action_type": "...", "column": "...", "fill_value": "...", "target_dtype": "...", "normalize_map": {}, "reference_table": "...", "reference_column": "...", "z_threshold": 3.0}"""


def run_baseline():
    """Main inference loop processing all tasks."""
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    for task_id, task in TASKS.items():
        # Track metrics for [END] block
        all_rewards = []
        total_steps = 0
        success = False
        
        # START block: [START] task=<task_name> env=<benchmark> model=<model_name>
        print(f"[START] task={task_id} env=data-cleaning model={MODEL_NAME}", flush=True)
        sys.stdout.flush()

        try:
            env = DataCleaningEnv()
            obs = env.reset(task_id)
            initial_df = env.original_df.copy()
            extra_tables = env._extra_tables.copy()
            history = []
            step_num = 0
            last_error = None

            for step_num in range(15):
                user_msg = f"""Dataset state:
- Rows: {obs.total_rows}, Nulls: {obs.null_count}, Duplicates: {obs.duplicate_count}, Quality: {obs.quality_score:.3f}
- Issues: {', '.join(obs.issues) if obs.issues else 'None'}
- Columns: {[f"{c.name}(nulls={c.null_count},dtype={c.dtype})" for c in obs.columns]}
Output a JSON action."""

                history.append({"role": "user", "content": user_msg})

                # Try to get LLM response
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
                    last_error = None
                except Exception as e:
                    last_error = str(e)
                    # Remove newlines from error message
                    last_error = last_error.replace('\n', ' ').replace('\r', ' ')
                    # [STEP] format: step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
                    error_field = f'"{last_error}"' if last_error else "null"
                    print(f"[STEP] step={step_num} action=none reward=0.00 done=true error={error_field}", flush=True)
                    sys.stdout.flush()
                    all_rewards.append(0.00)
                    total_steps = step_num + 1
                    break

                # Try to parse and execute action
                done = False
                reward_value = 0.00
                try:
                    action = Action(**json.loads(raw))
                    result = env.step(action)
                    obs = result.observation
                    reward_value = round(result.reward.value, 2)
                    done = result.done
                    last_error = None
                except Exception as e:
                    last_error = str(e)
                    # Remove newlines from error message
                    last_error = last_error.replace('\n', ' ').replace('\r', ' ')
                    reward_value = 0.00
                    done = False

                # [STEP] format: step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
                action_str = action.action_type if 'action' in locals() else 'parse_error'
                error_field = f'"{last_error}"' if last_error else "null"
                print(f"[STEP] step={step_num} action='{action_str}' reward={reward_value:.2f} done={'true' if done else 'false'} error={error_field}", flush=True)
                sys.stdout.flush()

                all_rewards.append(reward_value)
                total_steps = step_num + 1

                if done:
                    success = True
                    break

            # Try to grade the task
            try:
                result = run_grader(task_id, initial_df, env.df, extra_tables)
                if result.score >= 0.7:  # Assuming 0.7 is passing score
                    success = True
            except Exception as e:
                last_error = str(e)

        except Exception as e:
            last_error = str(e)
            total_steps = 0
            success = False

        # Format rewards list: r1,r2,...,rn (2 decimal places each)
        rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)

        # [END] format: success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
        print(f"[END] success={'true' if success else 'false'} steps={total_steps} rewards={rewards_str}", flush=True)
        sys.stdout.flush()


# Execute main logic immediately when module is loaded/imported
# (not just when run as __main__, in case validator imports this differently)
def main():
    try:
        # Check if imports failed
        if not IMPORTS_OK:
            error_msg = IMPORT_ERROR.replace('"', '\\"')
            print(f'[END] success=false steps=0 rewards= error="Import failed: {error_msg}"', flush=True)
            sys.stdout.flush()
            return
        
        # Check HF_TOKEN inside main block so we can print output even if it fails
        if HF_TOKEN is None:
            print('[END] success=false steps=0 rewards= error="HF_TOKEN environment variable is required"', flush=True)
            sys.stdout.flush()
            return
        
        run_baseline()
    except Exception as e:
        # Ensure we always print an [END] block even on fatal errors
        error_msg = str(e).replace('"', '\\"')  # Escape quotes in error message
        print(f'[END] success=false steps=0 rewards= error="{error_msg}"', flush=True)
        sys.stdout.flush()


# Run immediately when imported/executed
main()

if __name__ == "__main__":
    pass  # main() already ran above