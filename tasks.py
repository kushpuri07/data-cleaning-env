import pandas as pd
import numpy as np
from typing import Dict, Any
from models import GraderResult


#Task 1: Easy (Fix nulls and wrong dtypes)
def generate_easy_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "age":         [np.nan if i % 10 == 0 else np.random.randint(18, 70) for i in range(n)],
        "salary":      [np.nan if i % 8 == 0 else np.random.randint(30000, 120000) for i in range(n)],
        "email":       [np.nan if i % 12 == 0 else f"user{i}@example.com" for i in range(n)],
        "join_date":   [np.nan if i % 15 == 0 else f"202{i%4}-0{(i%9)+1}-01" for i in range(n)],
        "score":       [str(np.random.randint(0, 100)) for _ in range(n)],  # stored as str, should be int
    })
    return df


def grade_easy(df_initial: pd.DataFrame, df_final: pd.DataFrame) -> GraderResult:
    """
    Score based on:
    - 60%: nulls fixed
    - 40%: score column converted to numeric
    """
    initial_nulls = df_initial.isnull().sum().sum()
    final_nulls   = df_final.isnull().sum().sum()
    null_score    = 1.0 - (final_nulls / max(initial_nulls, 1))

    dtype_score = 1.0 if pd.api.types.is_numeric_dtype(df_final["score"]) else 0.0

    total = round(0.6 * null_score + 0.4 * dtype_score, 4)
    return GraderResult(
        task_id="task_easy",
        score=total,
        breakdown={"null_score": round(null_score, 4), "dtype_score": dtype_score},
        passed=total >= 0.7,
        explanation=(
            f"Nulls fixed: {initial_nulls - final_nulls}/{initial_nulls}. "
            f"Score column numeric: {bool(dtype_score)}."
        )
    )


#Task 2: Medium (Dedup + normalize strings)

def generate_medium_data() -> pd.DataFrame:
    np.random.seed(7)
    n = 120

    countries_raw = ["USA", "US", "United States", "uk", "UK", "United Kingdom",
                     "India", "india", "IN", "Germany", "DE", "germany"]
    statuses_raw  = ["active", "Active", "ACTIVE", "inactive", "Inactive", "INACTIVE",
                     "pending", "Pending", "PENDING"]

    df = pd.DataFrame({
        "order_id":  list(range(1, n + 1)),
        "customer":  [f"Customer_{np.random.randint(1, 30)}" for _ in range(n)],
        "country":   [np.random.choice(countries_raw) for _ in range(n)],
        "status":    [np.random.choice(statuses_raw)  for _ in range(n)],
        "amount":    [round(np.random.uniform(10, 500), 2) for _ in range(n)],
        "email":     [np.nan if i % 9 == 0 else f"cust{np.random.randint(1,30)}@mail.com" for i in range(n)],
    })

    # Inject duplicates (20 duplicate rows)
    dup_indices = np.random.choice(df.index, size=20, replace=True)
    dups = df.iloc[dup_indices].copy()
    df = pd.concat([df, dups], ignore_index=True)
    return df


def grade_medium(df_initial: pd.DataFrame, df_final: pd.DataFrame) -> GraderResult:
    """
    Score based on:
    - 40%: duplicates removed
    - 30%: country normalized (≤ 4 unique values)
    - 30%: status normalized (≤ 3 unique values)
    """
    initial_dups = df_initial.duplicated().sum()
    final_dups   = df_final.duplicated().sum()
    dup_score    = 1.0 - (final_dups / max(initial_dups, 1))

    country_unique = df_final["country"].str.strip().str.lower().nunique()
    country_score  = 1.0 if country_unique <= 4 else max(0.0, 1.0 - (country_unique - 4) * 0.15)

    status_unique = df_final["status"].str.strip().str.lower().nunique()
    status_score  = 1.0 if status_unique <= 3 else max(0.0, 1.0 - (status_unique - 3) * 0.2)

    total = round(0.4 * dup_score + 0.3 * country_score + 0.3 * status_score, 4)
    return GraderResult(
        task_id="task_medium",
        score=total,
        breakdown={
            "dup_score":     round(dup_score, 4),
            "country_score": round(country_score, 4),
            "status_score":  round(status_score, 4),
        },
        passed=total >= 0.7,
        explanation=(
            f"Duplicates remaining: {final_dups}/{initial_dups}. "
            f"Country variants: {country_unique} (target ≤4). "
            f"Status variants: {status_unique} (target ≤3)."
        )
    )


#Task 3: Hard (Broken FK + outliers + mixed encodings)

def generate_hard_data() -> pd.DataFrame:
    np.random.seed(99)
    n = 150

    # Reference table (valid product IDs)
    products = pd.DataFrame({
        "product_id":   list(range(1, 21)),
        "product_name": [f"Product_{i}" for i in range(1, 21)],
    })

    # Orders table with intentional issues
    valid_ids   = list(range(1, 21))
    invalid_ids = [99, 100, 101, 999, 888]  # broken FK references

    product_ids = [
        np.random.choice(invalid_ids) if i % 7 == 0
        else np.random.choice(valid_ids)
        for i in range(n)
    ]

    # Prices with outliers
    prices = [round(np.random.uniform(5, 200), 2) for _ in range(n)]
    for idx in np.random.choice(range(n), size=8, replace=False):
        prices[idx] = np.random.choice([9999.99, -50.0, 100000.0])  # outliers

    # Notes with encoding issues
    notes = [f"Order note {i}" for i in range(n)]
    bad_chars = ["\x80", "\x93", "\x94", "\xff"]
    for idx in np.random.choice(range(n), size=15, replace=False):
        notes[idx] = notes[idx] + np.random.choice(bad_chars)

    df = pd.DataFrame({
        "order_id":   range(1, n + 1),
        "product_id": product_ids,
        "price":      prices,
        "quantity":   [np.nan if i % 11 == 0 else np.random.randint(1, 20) for i in range(n)],
        "region":     [np.random.choice(["North", "South", "East", "West", "N", "S"]) for _ in range(n)],
        "notes":      notes,
    })

    return df


def get_hard_extra_tables() -> Dict[str, pd.DataFrame]:
    return {
        "products": pd.DataFrame({
            "product_id":   list(range(1, 21)),
            "product_name": [f"Product_{i}" for i in range(1, 21)],
        })
    }


def grade_hard(df_initial: pd.DataFrame, df_final: pd.DataFrame,
               extra_tables: Dict[str, pd.DataFrame]) -> GraderResult:
    """
    Score based on:
    - 30%: broken FK references fixed
    - 25%: price outliers removed
    - 25%: encoding issues fixed
    - 20%: nulls fixed
    """
    # FK score
    valid_ids     = set(extra_tables["products"]["product_id"].tolist())
    initial_fk_bad = (~df_initial["product_id"].isin(valid_ids)).sum()
    final_fk_bad   = (~df_final["product_id"].isin(valid_ids)).sum()
    fk_score       = 1.0 - (final_fk_bad / max(initial_fk_bad, 1))

    # Outlier score (prices between 0 and 1000 are reasonable)
    initial_outliers = ((df_initial["price"] < 0) | (df_initial["price"] > 1000)).sum()
    final_outliers   = ((df_final["price"]   < 0) | (df_final["price"]   > 1000)).sum()
    outlier_score    = 1.0 - (final_outliers / max(initial_outliers, 1))

    # Encoding score
    def count_bad_encoding(df):
        count = 0
        for val in df["notes"].dropna():
            try:
                val.encode("utf-8").decode("utf-8")
            except Exception:
                count += 1
            if any(c in val for c in ["\x80", "\x93", "\x94", "\xff"]):
                count += 1
        return count

    initial_enc = count_bad_encoding(df_initial)
    final_enc   = count_bad_encoding(df_final)
    enc_score   = 1.0 - (final_enc / max(initial_enc, 1))

    # Null score
    initial_nulls = df_initial.isnull().sum().sum()
    final_nulls   = df_final.isnull().sum().sum()
    null_score    = 1.0 - (final_nulls / max(initial_nulls, 1))

    total = round(
        0.30 * fk_score +
        0.25 * outlier_score +
        0.25 * enc_score +
        0.20 * null_score,
        4
    )

    return GraderResult(
        task_id="task_hard",
        score=total,
        breakdown={
            "fk_score":      round(fk_score, 4),
            "outlier_score": round(outlier_score, 4),
            "enc_score":     round(enc_score, 4),
            "null_score":    round(null_score, 4),
        },
        passed=total >= 0.7,
        explanation=(
            f"Bad FK refs: {final_fk_bad}/{initial_fk_bad} remaining. "
            f"Outliers: {final_outliers}/{initial_outliers} remaining. "
            f"Encoding issues: {final_enc}/{initial_enc} remaining. "
            f"Nulls: {final_nulls}/{initial_nulls} remaining."
        )
    )


#Task registry

TASKS: Dict[str, Any] = {
    "task_easy": {
        "task_id":     "task_easy",
        "name":        "Fix Nulls and Data Types",
        "description": "A customer dataset with missing values and a column stored as the wrong type. "
                       "Fill nulls with appropriate values and cast the score column to numeric.",
        "difficulty":  "easy",
        "max_steps":   20,
        "generate_data":  generate_easy_data,
        "grade":          lambda df_init, df_final, _extras: grade_easy(df_init, df_final),
        "extra_tables":   {},
    },
    "task_medium": {
        "task_id":     "task_medium",
        "name":        "Deduplicate and Normalize Strings",
        "description": "An orders dataset with duplicate rows and inconsistent string values "
                       "(country names and statuses). Remove duplicates and normalize to canonical forms.",
        "difficulty":  "medium",
        "max_steps":   30,
        "generate_data":  generate_medium_data,
        "grade":          lambda df_init, df_final, _extras: grade_medium(df_init, df_final),
        "extra_tables":   {},
    },
    "task_hard": {
        "task_id":     "task_hard",
        "name":        "Fix FK Violations, Outliers, and Encoding",
        "description": "A multi-issue orders dataset with broken foreign key references to a products table, "
                       "extreme price outliers, corrupted character encoding in notes, and missing quantities.",
        "difficulty":  "hard",
        "max_steps":   50,
        "generate_data":  generate_hard_data,
        "grade":          grade_hard,
        "extra_tables":   get_hard_extra_tables(),
    },
}


#Helper used by app.py

def run_grader(task_id: str, df_initial: pd.DataFrame,
               df_final: pd.DataFrame, extra_tables: Dict[str, pd.DataFrame]) -> GraderResult:
    task = TASKS.get(task_id)
    if not task:
        raise ValueError(f"Unknown task_id: {task_id}")
    return task["grade"](df_initial, df_final, extra_tables)