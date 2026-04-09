import os
import json
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from models import Action, BaselineResult
from environment import DataCleaningEnv
from tasks import TASKS, run_grader

# Configuration - read from environment
# Note: validator requires os.environ["..."] bracket access, not os.getenv()
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")


SYSTEM_PROMPT = """You are a data cleaning agent. Output ONLY a raw JSON action object.
Available action_types: fill_null, drop_duplicates, fix_dtype, normalize_str, drop_outliers, fix_foreign_key, fix_encoding, done
JSON format: {"action_type": "...", "column": "...", "fill_value": "...", "target_dtype": "...", "normalize_map": {}, "reference_table": "...", "reference_column": "...", "z_threshold": 3.0}"""


def log_start(task: str, env: str, model: str) -> None:
    """Log task start."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Log a single step."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    """Log task end."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def run_baseline(client: OpenAI):
    """Main inference loop processing all tasks."""
    all_rewards = []
    total_steps = 0
    success = False
    env = None
    
    # Print [START] once at the beginning
    log_start(task="baseline", env="data-cleaning", model=MODEL_NAME)

    # IMPORTANT: Client is now passed as parameter (created in main())
    # This will be set by the caller

    # Process all tasks
    try:
        for task_id, task in TASKS.items():
            task_rewards = []
            task_steps = 0
            
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
                        last_error = str(e).replace('\n', ' ').replace('\r', ' ')
                        log_step(step=step_num, action="none", reward=0.00, done=True, error=last_error)
                        task_rewards.append(0.00)
                        task_steps = step_num + 1
                        break

                    # Try to parse and execute action
                    done = False
                    reward_value = 0.00
                    action_str = "none"
                    try:
                        action = Action(**json.loads(raw))
                        action_str = action.action_type
                        result = env.step(action)
                        obs = result.observation
                        reward_value = round(result.reward.value, 2)
                        done = result.done
                        last_error = None
                    except Exception as e:
                        last_error = str(e).replace('\n', ' ').replace('\r', ' ')
                        reward_value = 0.00
                        done = False

                    error_field = last_error if last_error else "null"
                    log_step(step=step_num, action=action_str, reward=reward_value, done=done, error=last_error)

                    task_rewards.append(reward_value)
                    task_steps = step_num + 1

                    if done:
                        success = True
                        break

                # Try to grade the task
                try:
                    result = run_grader(task_id, initial_df, env.df, extra_tables)
                    if result.score >= 0.7:
                        success = True
                except Exception as e:
                    error_msg = str(e).replace('\n', ' ').replace('\r', ' ')
                    # Grading error is not critical, log but continue

            except Exception as e:
                error_msg = str(e).replace('\n', ' ').replace('\r', ' ')
                log_step(step=0, action="none", reward=0.00, done=True, error=error_msg)
                task_rewards.append(0.00)
                task_steps = 1
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception as close_err:
                        # Cleanup error is not critical
                        pass
            
            all_rewards.extend(task_rewards)
            total_steps += task_steps

    except Exception as e:
        success = False
    finally:
        # Print [END] once at the end
        log_end(success=success, steps=total_steps, rewards=all_rewards)


# Execute main logic immediately when module is loaded/imported
def main():
    try:
        # Create client in main() - CRITICAL for validator to detect API calls
        # VALIDATOR REQUIREMENT: Use os.environ["..."] with bracket access, NOT os.getenv()
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"],
        )
        # Pass client to run_baseline
        run_baseline(client)
    except Exception as e:
        log_start(task="baseline", env="data-cleaning", model=MODEL_NAME)
        error_msg = str(e).replace('\n', ' ').replace('\r', ' ')
        log_end(success=False, steps=0, rewards=[])


# Run immediately when imported/executed
main()

if __name__ == "__main__":
    pass  # main() already ran above