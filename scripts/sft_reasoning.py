from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_reasoning_csv(path: str | Path) -> pd.DataFrame:
    """
    Загружает reasoning dataset для Reasoning-SFT.

    Ожидаемый формат:
    - equation
    - answer
    - type
    - prompt
    - expert_trajectory

    target для обучения = expert_trajectory
    """
    path = Path(path)
    df = pd.read_csv(path)

    required_cols = {
        "equation",
        "answer",
        "type",
        "prompt",
        "expert_trajectory",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.copy()

    df["equation"] = df["equation"].astype(str)
    df["answer"] = df["answer"].astype(str)
    df["type"] = df["type"].astype(str).str.strip()
    df["prompt"] = df["prompt"].astype(str)
    df["expert_trajectory"] = df["expert_trajectory"].astype(str)

    # Главное отличие reasoning-SFT:
    # обучаем модель не на ответе, а на полной expert trajectory.
    df["target"] = df["expert_trajectory"]

    # Удобно для проверки/debug.
    df["full_text"] = df["prompt"] + "\n" + df["target"]

    # Мини-проверки качества датасета.
    empty_target = df["target"].str.strip() == ""
    if empty_target.any():
        raise ValueError(f"{path} has empty expert_trajectory rows: {empty_target.sum()}")

    no_action = ~df["target"].str.contains("ACTION 1:", regex=False)
    if no_action.any():
        raise ValueError(f"{path} has rows without ACTION 1: {no_action.sum()}")

    no_final = ~df["target"].str.contains("FINAL:", regex=False)
    if no_final.any():
        raise ValueError(f"{path} has rows without FINAL: {no_final.sum()}")

    no_boxed = ~df["target"].str.contains("\\boxed", regex=False)
    if no_boxed.any():
        raise ValueError(f"{path} has rows without \\boxed: {no_boxed.sum()}")

    return df.reset_index(drop=True)


def load_reasoning_train_val_dataframes(
    train_path: str | Path,
    val_path: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = load_reasoning_csv(train_path)
    val_df = load_reasoning_csv(val_path)
    return train_df, val_df
