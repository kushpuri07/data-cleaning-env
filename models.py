from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from enum import Enum


#Action types the agent can perform

class ActionType(str, Enum):
    FILL_NULL       = "fill_null"        # Fill missing values in a column
    DROP_DUPLICATES = "drop_duplicates"  # Remove duplicate rows
    FIX_DTYPE       = "fix_dtype"        # Cast a column to the correct type
    NORMALIZE_STR   = "normalize_str"    # Standardize inconsistent strings
    DROP_OUTLIERS   = "drop_outliers"    # Remove statistical outliers
    FIX_FOREIGN_KEY = "fix_foreign_key"  # Repair broken FK references
    FIX_ENCODING    = "fix_encoding"     # Fix mixed/corrupt character encoding
    DONE            = "done"             # Agent signals episode is complete


#Action

class Action(BaseModel):
    """What the agent sends to step()."""

    action_type: ActionType = Field(
        ...,
        description="The cleaning operation to perform."
    )
    column: Optional[str] = Field(
        None,
        description="Target column name (required for most actions)."
    )
    fill_value: Optional[Any] = Field(
        None,
        description="Value to use when action_type is fill_null. "
                    "Use 'mean', 'median', 'mode', or a literal value."
    )
    target_dtype: Optional[str] = Field(
        None,
        description="Target dtype string for fix_dtype, e.g. 'int', 'float', 'str', 'datetime'."
    )
    normalize_map: Optional[Dict[str, str]] = Field(
        None,
        description="Mapping of raw → canonical strings for normalize_str, "
                    "e.g. {'USA': 'United States', 'US': 'United States'}."
    )
    reference_table: Optional[str] = Field(
        None,
        description="Name of the reference table for fix_foreign_key."
    )
    reference_column: Optional[str] = Field(
        None,
        description="Column in the reference table to validate against."
    )
    z_threshold: Optional[float] = Field(
        3.0,
        description="Z-score threshold for drop_outliers (default 3.0)."
    )

    model_config = {"use_enum_values": True}


#Observation

class ColumnSummary(BaseModel):
    """Per-column snapshot the agent sees."""
    name:          str
    dtype:         str
    null_count:    int
    unique_count:  int
    sample_values: List[Any]   # up to 5 representative values


class Observation(BaseModel):
    """What step() and reset() return for the agent to see."""

    task_id:         str   = Field(..., description="Which task is running.")
    step_number:     int   = Field(..., description="How many steps taken so far.")
    total_rows:      int   = Field(..., description="Current row count.")
    total_columns:   int   = Field(..., description="Current column count.")
    columns:         List[ColumnSummary]
    null_count:      int   = Field(..., description="Total nulls remaining across all columns.")
    duplicate_count: int   = Field(..., description="Duplicate rows remaining.")
    quality_score:   float = Field(..., description="Current data quality score 0.0–1.0.")
    issues:          List[str] = Field(
        default_factory=list,
        description="Human-readable list of detected issues (for agent context)."
    )
    info:            Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra task-specific metadata."
    )


#Reward

class Reward(BaseModel):
    """Returned alongside Observation from step()."""

    value:       float = Field(..., description="Reward for this step (-1.0 to +1.0).")
    breakdown:   Dict[str, float] = Field(
        default_factory=dict,
        description="Per-component reward breakdown for interpretability."
    )
    explanation: str   = Field("", description="Human-readable reason for this reward.")


#Step response

class StepResponse(BaseModel):
    """Full response from POST /step."""
    observation: Observation
    reward:      Reward
    done:        bool  = Field(..., description="True when the episode has ended.")
    info:        Dict[str, Any] = Field(default_factory=dict)


#State

class EnvironmentState(BaseModel):
    """Returned by GET /state — full internal snapshot."""
    task_id:           str
    step_number:       int
    max_steps:         int
    initial_issues:    Dict[str, int]   # e.g. {"nulls": 42, "duplicates": 7}
    remaining_issues:  Dict[str, int]
    quality_score:     float
    episode_reward:    float            # cumulative reward so far
    done:              bool


#Task metadata

class TaskInfo(BaseModel):
    """Returned by GET /tasks."""
    task_id:     str
    name:        str
    description: str
    difficulty:  str   # "easy" | "medium" | "hard"
    max_steps:   int
    action_schema: Dict[str, Any]  # JSON schema of Action model


#Grader result

class GraderResult(BaseModel):
    """Returned by GET /grader after an episode."""
    task_id:    str
    score:      float = Field(..., description="Final score 0.0–1.0.")
    breakdown:  Dict[str, float]
    passed:     bool
    explanation: str


#Baseline result

class BaselineResult(BaseModel):
    """Returned by GET /baseline."""
    model:   str
    results: Dict[str, float]   # task_id → score
    average: float