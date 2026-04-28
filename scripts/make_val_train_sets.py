import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def validate_columns(df: pd.DataFrame, path: Path) -> None:
    required = {"equation", "answer", "type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В файле {path} нет колонок: {sorted(missing)}")


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["equation"] = df["equation"].astype(str)
    df["answer"] = df["answer"].astype(str)
    df["type"] = df["type"].astype(str).str.strip()
    return df


def make_balanced_val_for_metrics(
    val_df: pd.DataFrame,
    n_per_type: int,
    seed: int,
) -> pd.DataFrame:
    parts = []

    for eq_type, group in val_df.groupby("type"):
        n = min(len(group), n_per_type)
        sampled = group.sample(n=n, random_state=seed)
        parts.append(sampled)

    balanced_df = pd.concat(parts, axis=0)
    balanced_df = balanced_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced_df


def save_manifest(
    out_path: Path,
    config: dict,
    train_big: pd.DataFrame,
    val_big: pd.DataFrame,
    balanced_val_metrics: pd.DataFrame,
) -> None:
    manifest = {
        "config": config,
        "train_big": {
            "size": len(train_big),
            "distribution": train_big["type"].value_counts().sort_index().to_dict(),
            "purpose": "Большой train для основных SFT/RL экспериментов.",
        },
        "val_big": {
            "size": len(val_big),
            "distribution": val_big["type"].value_counts().sort_index().to_dict(),
            "purpose": "Большой validation split из исходного train.csv.",
        },
        "balanced_val_metrics_big": {
            "size": len(balanced_val_metrics),
            "distribution": balanced_val_metrics["type"].value_counts().sort_index().to_dict(),
            "purpose": "Фиксированная balanced validation выборка для быстрых метрик во время обучения.",
        },
        "test": {
            "path": "data/test.csv",
            "purpose": "Финальный test. Скрипт его не изменяет.",
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main():
    SEED = 42
    VAL_SIZE = 0.2
    N_VAL_METRICS_PER_TYPE = 20

    DATA_DIR = Path("data")

    TRAIN_SOURCE = DATA_DIR / "train.csv"
    TEST_SOURCE = DATA_DIR / "test.csv"  # не трогаем

    OUT_TRAIN_BIG = DATA_DIR / "train_big.csv"
    OUT_VAL_BIG = DATA_DIR / "val_big.csv"
    OUT_BALANCED_VAL_METRICS = DATA_DIR / "balanced_val_metrics_big.csv"
    OUT_MANIFEST = DATA_DIR / "big_split_manifest.json"

    config = {
        "seed": SEED,
        "val_size": VAL_SIZE,
        "n_val_metrics_per_type": N_VAL_METRICS_PER_TYPE,
        "train_source": str(TRAIN_SOURCE),
        "test_source": str(TEST_SOURCE),
        "out_train_big": str(OUT_TRAIN_BIG),
        "out_val_big": str(OUT_VAL_BIG),
        "out_balanced_val_metrics_big": str(OUT_BALANCED_VAL_METRICS),
        "note": "test.csv is not touched; train_big and val_big are created only from train.csv.",
    }

    if not TRAIN_SOURCE.exists():
        raise FileNotFoundError(f"Не найден файл: {TRAIN_SOURCE}")

    df = pd.read_csv(TRAIN_SOURCE)
    validate_columns(df, TRAIN_SOURCE)
    df = normalize_df(df)

    print(f"Загружен {TRAIN_SOURCE}: {len(df)} строк")
    print("\nИсходное распределение type:")
    print(df["type"].value_counts())

    train_big, val_big = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=df["type"],
        shuffle=True,
    )

    train_big = train_big.reset_index(drop=True)
    val_big = val_big.reset_index(drop=True)

    balanced_val_metrics = make_balanced_val_for_metrics(
        val_df=val_big,
        n_per_type=N_VAL_METRICS_PER_TYPE,
        seed=SEED,
    )

    train_big.to_csv(OUT_TRAIN_BIG, index=False)
    val_big.to_csv(OUT_VAL_BIG, index=False)
    balanced_val_metrics.to_csv(OUT_BALANCED_VAL_METRICS, index=False)

    save_manifest(
        out_path=OUT_MANIFEST,
        config=config,
        train_big=train_big,
        val_big=val_big,
        balanced_val_metrics=balanced_val_metrics,
    )

    print("\nФайлы сохранены:")
    print(f"  train_big:                {OUT_TRAIN_BIG} ({len(train_big)} строк)")
    print(f"  val_big:                  {OUT_VAL_BIG} ({len(val_big)} строк)")
    print(f"  balanced_val_metrics_big: {OUT_BALANCED_VAL_METRICS} ({len(balanced_val_metrics)} строк)")
    print(f"  manifest:                 {OUT_MANIFEST}")

    print("\nРаспределение train_big:")
    print(train_big["type"].value_counts())

    print("\nРаспределение val_big:")
    print(val_big["type"].value_counts())

    print("\nРаспределение balanced_val_metrics_big:")
    print(balanced_val_metrics["type"].value_counts())

    print("\nВажно: data/test.csv не изменялся.")


if __name__ == "__main__":
    main()
