import os
import json
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from models import (
    Action, Observation, StepResponse, EnvironmentState,
    TaskInfo, GraderResult, BaselineResult
)
from environment import DataCleaningEnv
from tasks import TASKS, run_grader


#App setup

app = FastAPI(
    title="Data Cleaning OpenEnv",
    description="An OpenEnv-compliant RL environment for training agents to clean messy datasets.",
    version="1.0.0",
)

# One env instance per server (stateful — single agent at a time)
env = DataCleaningEnv()

# Store initial df for grading at end of episode
_initial_df_store: Dict[str, Any] = {}


#Request bodies

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_easy"


#Health check

@app.get("/health")
def health():
    """Ping endpoint — must return 200 for HF Space validation."""
    return {"status": "ok", "env": "data-cleaning-openenv"}


#reset
@app.post("/reset", response_model=Observation)
def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()
    """
    Start a new episode.
    Returns the initial observation (dirty dataset summary).
    """
    if req.task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {req.task_id}")

    obs = env.reset(req.task_id)

    # Save initial df snapshot for grader
    _initial_df_store["df"]           = env.original_df.copy()
    _initial_df_store["task_id"]      = req.task_id
    _initial_df_store["extra_tables"] = env._extra_tables.copy()

    return obs


#step

@app.post("/step", response_model=StepResponse)
def step(action: Action):
    """
    Apply one cleaning action.
    Returns observation, reward, done flag, and info.
    """
    if env.df is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    if env.done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset to start a new one.")

    try:
        result = env.step(action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    return result


#state

@app.get("/state", response_model=EnvironmentState)
def state():
    """Return the current internal state of the environment."""
    if env.df is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return env.state()


#tasks

@app.get("/tasks")
def list_tasks():
    """
    List all available tasks and the action schema.
    Required by the OpenEnv spec.
    """
    task_list = []
    for tid, task in TASKS.items():
        task_list.append(TaskInfo(
            task_id=tid,
            name=task["name"],
            description=task["description"],
            difficulty=task["difficulty"],
            max_steps=task["max_steps"],
            action_schema=Action.model_json_schema(),
        ).model_dump())

    return {"tasks": task_list, "action_schema": Action.schema()}


#grader

@app.get("/grader", response_model=GraderResult)
def grader():
    """
    Score the current (or just-finished) episode.
    Returns a deterministic 0.0–1.0 score with breakdown.
    """
    if not _initial_df_store.get("df") is not None:
        raise HTTPException(status_code=400, detail="No episode data. Call /reset then play an episode.")
    if env.df is None:
        raise HTTPException(status_code=400, detail="No active episode data.")

    try:
        result = run_grader(
            task_id=_initial_df_store["task_id"],
            df_initial=_initial_df_store["df"],
            df_final=env.df,
            extra_tables=_initial_df_store.get("extra_tables", {}),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


#baseline

@app.get("/baseline", response_model=BaselineResult)
def baseline():
    """
    Run the baseline inference script against all 3 tasks.
    Returns reproducible scores for each task.
    Reads OPENAI_API_KEY from environment variables.
    """
    try:
        from inference import run_baseline
        result = run_baseline()
        return result
    except ImportError:
        raise HTTPException(status_code=500, detail="baseline.py not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Baseline failed: {str(e)}")


#root

@app.get("/")
def root():
    return {
        "name":        "Data Cleaning OpenEnv",
        "version":     "1.0.0",
        "description": "OpenEnv environment for training AI agents to clean messy datasets.",
        "endpoints": {
            "POST /reset":    "Start a new episode",
            "POST /step":     "Apply a cleaning action",
            "GET  /state":    "Get current environment state",
            "GET  /tasks":    "List tasks and action schema",
            "GET  /grader":   "Get episode score",
            "GET  /baseline": "Run baseline agent on all tasks",
            "GET  /health":   "Health check",
        }
    }


#entrypoint

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)