import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Tuple
from models import (
    Action, ActionType, Observation, Reward, StepResponse,
    EnvironmentState, ColumnSummary
)


class DataCleaningEnv:
    """
    OpenEnv-compliant environment for data cleaning tasks.

    The agent interacts via:
        reset(task_id)  → Observation
        step(action)    → StepResponse
        state()         → EnvironmentState
    """

    MAX_STEPS = {
        "task_easy":   20,
        "task_medium": 30,
        "task_hard":   50,
    }

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.original_df: Optional[pd.DataFrame] = None
        self.task_id: str = ""
        self.step_number: int = 0
        self.episode_reward: float = 0.0
        self.done: bool = False
        self.initial_issues: Dict[str, int] = {}
        self._extra_tables: Dict[str, pd.DataFrame] = {}  # for FK tasks

    #reset

    def reset(self, task_id: str) -> Observation:
        """Start a new episode for the given task."""
        from tasks import TASKS
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")

        task = TASKS[task_id]
        self.df = task["generate_data"]()
        self.original_df = self.df.copy()
        self._extra_tables = task.get("extra_tables", {})
        self.task_id = task_id
        self.step_number = 0
        self.episode_reward = 0.0
        self.done = False
        self.initial_issues = self._count_issues()
        return self._make_observation()

    #step

    def step(self, action: Action) -> StepResponse:
        """Apply one cleaning action and return the result."""
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self.step_number += 1
        max_steps = self.MAX_STEPS.get(self.task_id, 30)

        if action.action_type == ActionType.DONE:
            reward = self._compute_final_reward()
            self.done = True
        else:
            reward = self._apply_action(action)

        # End episode if max steps reached
        if self.step_number >= max_steps:
            self.done = True

        self.episode_reward += reward.value
        obs = self._make_observation()
        return StepResponse(
            observation=obs,
            reward=reward,
            done=self.done,
            info={"step": self.step_number, "max_steps": max_steps}
        )

    #state

    def state(self) -> EnvironmentState:
        return EnvironmentState(
            task_id=self.task_id,
            step_number=self.step_number,
            max_steps=self.MAX_STEPS.get(self.task_id, 30),
            initial_issues=self.initial_issues,
            remaining_issues=self._count_issues(),
            quality_score=self._quality_score(),
            episode_reward=self.episode_reward,
            done=self.done,
        )

    # ── action dispatcher ──────────────────────────────────────────────────

    def _apply_action(self, action: Action) -> Reward:
        try:
            if action.action_type == ActionType.FILL_NULL:
                return self._fill_null(action)
            elif action.action_type == ActionType.DROP_DUPLICATES:
                return self._drop_duplicates()
            elif action.action_type == ActionType.FIX_DTYPE:
                return self._fix_dtype(action)
            elif action.action_type == ActionType.NORMALIZE_STR:
                return self._normalize_str(action)
            elif action.action_type == ActionType.DROP_OUTLIERS:
                return self._drop_outliers(action)
            elif action.action_type == ActionType.FIX_FOREIGN_KEY:
                return self._fix_foreign_key(action)
            elif action.action_type == ActionType.FIX_ENCODING:
                return self._fix_encoding(action)
            else:
                return Reward(value=-0.05, explanation="Unknown action type.")
        except Exception as e:
            return Reward(value=-0.1, explanation=f"Action failed: {str(e)}")

    # ── individual action handlers ─────────────────────────────────────────

    def _fill_null(self, action: Action) -> Reward:
        col = action.column
        if col not in self.df.columns:
            return Reward(value=-0.05, breakdown={}, explanation=f"Column '{col}' not found.")

        nulls_before = self.df[col].isnull().sum()
        if nulls_before == 0:
            return Reward(value=-0.05, breakdown={}, explanation="No nulls to fill in this column.")

        fv = action.fill_value
        if fv == "mean":
            fill = self.df[col].mean()
        elif fv == "median":
            fill = self.df[col].median()
        elif fv == "mode":
            fill = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else None
        else:
            fill = fv

        self.df[col] = self.df[col].fillna(fill)
        nulls_after = self.df[col].isnull().sum()
        fixed = nulls_before - nulls_after
        reward_val = round(0.1 * fixed, 3)
        return Reward(
            value=reward_val,
            breakdown={"nulls_fixed": fixed},
            explanation=f"Filled {fixed} nulls in '{col}' with {fv}."
        )

    def _drop_duplicates(self) -> Reward:
        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        removed = before - len(self.df)
        if removed == 0:
            return Reward(value=-0.05, breakdown={}, explanation="No duplicates found.")
        reward_val = round(0.15 * removed, 3)
        return Reward(
            value=reward_val,
            breakdown={"duplicates_removed": removed},
            explanation=f"Removed {removed} duplicate rows."
        )

    def _fix_dtype(self, action: Action) -> Reward:
        col = action.column
        dtype = action.target_dtype
        if col not in self.df.columns:
            return Reward(value=-0.05, breakdown={}, explanation=f"Column '{col}' not found.")
        try:
            if dtype == "int":
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").astype("Int64")
            elif dtype == "float":
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
            elif dtype == "str":
                self.df[col] = self.df[col].astype(str)
            elif dtype == "datetime":
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
            else:
                return Reward(value=-0.05, breakdown={}, explanation=f"Unknown dtype: {dtype}")
            return Reward(value=0.15, breakdown={"dtype_fixed": 1},
                          explanation=f"Cast '{col}' to {dtype}.")
        except Exception as e:
            return Reward(value=-0.1, breakdown={}, explanation=str(e))

    def _normalize_str(self, action: Action) -> Reward:
        col = action.column
        mapping = action.normalize_map or {}
        if col not in self.df.columns:
            return Reward(value=-0.05, breakdown={}, explanation=f"Column '{col}' not found.")
        before = self.df[col].nunique()
        self.df[col] = self.df[col].replace(mapping)
        # Also strip whitespace and unify case
        self.df[col] = self.df[col].astype(str).str.strip()
        after = self.df[col].nunique()
        fixed = before - after
        if fixed <= 0:
            return Reward(value=-0.02, breakdown={}, explanation="Normalization had no effect.")
        return Reward(value=round(0.1 * fixed, 3), breakdown={"variants_collapsed": fixed},
                      explanation=f"Collapsed {fixed} string variants in '{col}'.")

    def _drop_outliers(self, action: Action) -> Reward:
        col = action.column
        z = action.z_threshold or 3.0
        if col not in self.df.columns:
            return Reward(value=-0.05, breakdown={}, explanation=f"Column '{col}' not found.")
        numeric = pd.to_numeric(self.df[col], errors="coerce")
        mean, std = numeric.mean(), numeric.std()
        if std == 0:
            return Reward(value=-0.02, breakdown={}, explanation="No variance to compute outliers.")
        mask = ((numeric - mean) / std).abs() <= z
        removed = (~mask).sum()
        self.df = self.df[mask].reset_index(drop=True)
        return Reward(value=round(0.12 * removed, 3), breakdown={"outliers_removed": int(removed)},
                      explanation=f"Removed {removed} outliers from '{col}' (z>{z}).")

    def _fix_foreign_key(self, action: Action) -> Reward:
        col = action.column
        ref_table = action.reference_table
        ref_col = action.reference_column
        if col not in self.df.columns:
            return Reward(value=-0.05, breakdown={}, explanation=f"Column '{col}' not found.")
        if ref_table not in self._extra_tables:
            return Reward(value=-0.05, breakdown={}, explanation=f"Reference table '{ref_table}' not found.")
        valid = set(self._extra_tables[ref_table][ref_col].dropna().unique())
        invalid_mask = ~self.df[col].isin(valid) & self.df[col].notna()
        count = invalid_mask.sum()
        self.df.loc[invalid_mask, col] = None  # nullify broken FK refs
        return Reward(value=round(0.2 * count, 3), breakdown={"fk_fixed": int(count)},
                      explanation=f"Nullified {count} broken FK references in '{col}'.")

    def _fix_encoding(self, action: Action) -> Reward:
        col = action.column
        if col not in self.df.columns:
            return Reward(value=-0.05, breakdown={}, explanation=f"Column '{col}' not found.")
        fixed = 0
        def clean(v):
            nonlocal fixed
            if isinstance(v, str):
                cleaned = v.encode("utf-8", errors="ignore").decode("utf-8")
                if cleaned != v:
                    fixed += 1
                return cleaned
            return v
        self.df[col] = self.df[col].apply(clean)
        return Reward(value=round(0.1 * fixed, 3), breakdown={"encoding_fixed": fixed},
                      explanation=f"Fixed {fixed} encoding issues in '{col}'.")

    def _compute_final_reward(self) -> Reward:
        score = self._quality_score()
        val = round((score - 0.5) * 0.4, 3)  # bonus if score > 0.5, penalty if below
        return Reward(value=val, breakdown={"final_quality": score},
                      explanation=f"Episode ended. Final quality score: {score:.2f}.")

    # ── helpers ────────────────────────────────────────────────────────────

    def _count_issues(self) -> Dict[str, int]:
        if self.df is None:
            return {}
        return {
            "nulls":      int(self.df.isnull().sum().sum()),
            "duplicates": int(self.df.duplicated().sum()),
            "rows":       len(self.df),
        }

    def _quality_score(self) -> float:
        """0.0 = completely broken, 1.0 = perfectly clean."""
        if self.df is None or self.original_df is None:
            return 0.0
        total_cells = self.original_df.size or 1
        null_penalty  = self.df.isnull().sum().sum() / total_cells
        dup_penalty   = self.df.duplicated().sum() / max(len(self.df), 1)
        score = 1.0 - (0.6 * null_penalty + 0.4 * dup_penalty)
        return round(max(0.0, min(1.0, score)), 4)

    def _make_observation(self) -> Observation:
        columns = []
        for col in self.df.columns:
            sample = self.df[col].dropna().head(5).tolist()
            columns.append(ColumnSummary(
                name=col,
                dtype=str(self.df[col].dtype),
                null_count=int(self.df[col].isnull().sum()),
                unique_count=int(self.df[col].nunique()),
                sample_values=sample,
            ))

        issues = []
        issues_map = self._count_issues()
        if issues_map.get("nulls", 0):
            issues.append(f"{issues_map['nulls']} null values across columns.")
        if issues_map.get("duplicates", 0):
            issues.append(f"{issues_map['duplicates']} duplicate rows.")

        return Observation(
            task_id=self.task_id,
            step_number=self.step_number,
            total_rows=len(self.df),
            total_columns=len(self.df.columns),
            columns=columns,
            null_count=issues_map.get("nulls", 0),
            duplicate_count=issues_map.get("duplicates", 0),
            quality_score=self._quality_score(),
            issues=issues,
            info={},
        )