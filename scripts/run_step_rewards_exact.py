from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
import torch
from tqdm import tqdm

from sft import load_train_model_and_tokenizer
from metrics_utils import compute_metrics

from reward_utils import (
    action_exact_reward,
    build_next_step_prompt,
    extract_action,
    normalize_action,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "experiments/reasoning_sft/qwen25_math_15b_balanced_train_small_reasoning_v1_run01/final_model"
MODEL_SHORT_NAME = "qwen25_math_15b"

STEP_DATA_PATH = Path("data/step_data/balanced_val_small_steps_v1.csv")

OUTPUT_DIR = Path("experiments/step_rewards_exact/qwen25_math_15b_reasoning_v1_run01")

MLFLOW_EXPERIMENT = "ode_llm_diploma"
RUN_NAME = "step_rewards_exact_qwen25_math_15b_reasoning_v1_run01"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE_NAME = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"

BATCH_SIZE = 8
MAX_NEW_TOKENS = 512
SYMPY_TIMEOUT_SEC = 8


# ============================================================
# Helpers
# ============================================================

def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_step_data(df: pd.DataFrame) -> None:
    required = {
        "example_id",
        "step_id",
        "equation",
        "type",
        "previous_steps",
        "step_prompt",          # <-- добавить
        "target_step",
        "target_action",
        "answer",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Step dataset missing columns: {sorted(missing)}")


def generate_model_steps(df: pd.DataFrame, model, tokenizer) -> pd.DataFrame:
    work_df = df.copy()
    prompts = work_df["step_prompt"].tolist()         # берём готовые промпты
    model.eval()
    tokenizer.padding_side = "left"
    all_outputs = []

    for start in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[start:start + BATCH_SIZE]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(DEVICE)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, prompt_len:]
        batch_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        all_outputs.extend([text.strip() for text in batch_texts])

    work_df["model_step"] = all_outputs
    return work_df


def compute_step_rewards(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.copy()

    df["target_action_norm"] = df["target_action"].apply(normalize_action)
    df["model_action"] = df["model_step"].apply(extract_action)

    df["step_reward_exact"] = df.apply(
        lambda row: action_exact_reward(
            model_step=row["model_step"],
            target_action=row["target_action"],
        ),
        axis=1,
    )

    return df


def compute_final_sympy_rewards(step_reward_df: pd.DataFrame) -> pd.DataFrame:
    """
    SymPy reward считаем только на FINAL-строках.

    Для FINAL-строки:
    previous_steps = все предыдущие expert steps
    model_step = модельный FINAL
    answer = правильный ответ

    final_reward = 1 если SymPy проверка проходит, иначе 0.
    """
    final_df = step_reward_df[
        step_reward_df["target_action_norm"] == "final_answer"
    ].copy()

    if final_df.empty:
        print("[warning] Нет FINAL-строк для SymPy reward")
        step_reward_df["final_reward_sympy"] = None
        return step_reward_df

    metric_input = final_df.copy()
    metric_input["llm_solution"] = metric_input["model_step"]

    evaluated_df, metrics_df, class_df = compute_metrics(
        data=metric_input,
        porog=0.5,
        progress_log_path=None,
        log_every=10,
        partial_eval_save_path=None,
        save_every=25,
        compute_reference=False,
        timeout_sec=SYMPY_TIMEOUT_SEC,
    )

    # Пытаемся найти колонку с bool-результатом.
    possible_bool_cols = [
        "sympy_bool",
        "is_correct",
        "correct",
        "passed",
        "strict_correct",
    ]

    bool_col = None
    for col in possible_bool_cols:
        if col in evaluated_df.columns:
            bool_col = col
            break

    if bool_col is None:
        raise ValueError(
            f"Не нашла колонку SymPy bool в evaluated_df. Колонки: {list(evaluated_df.columns)}"
        )

    final_rewards = evaluated_df[["example_id", bool_col]].copy()
    final_rewards = final_rewards.rename(columns={bool_col: "final_reward_sympy"})
    final_rewards["final_reward_sympy"] = final_rewards["final_reward_sympy"].astype(int)

    merged = step_reward_df.merge(
        final_rewards,
        on="example_id",
        how="left",
    )

    # Для не-FINAL шагов final_reward оставляем пустым.
    merged.loc[
        merged["target_action_norm"] != "final_answer",
        "final_reward_sympy",
    ] = None

    return merged


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rows.append({
        "metric": "mean_step_reward_exact",
        "value": float(df["step_reward_exact"].mean()),
    })

    rows.append({
        "metric": "action_exact_match_rate",
        "value": float(df["step_reward_exact"].mean()),
    })

    final_mask = df["target_action_norm"] == "final_answer"
    if final_mask.any() and "final_reward_sympy" in df.columns:
        rows.append({
            "metric": "mean_final_reward_sympy",
            "value": float(df.loc[final_mask, "final_reward_sympy"].dropna().mean()),
        })

    rows.append({
        "metric": "num_step_rows",
        "value": int(len(df)),
    })

    rows.append({
        "metric": "num_final_rows",
        "value": int(final_mask.sum()),
    })

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "model_path": MODEL_PATH,
        "step_data_path": str(STEP_DATA_PATH),
        "output_dir": str(OUTPUT_DIR),
        "step_reward": "exact_action_match_0_1",
        "final_reward": "sympy_final_answer_0_1",
        "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "sympy_timeout_sec": SYMPY_TIMEOUT_SEC,
        "device": DEVICE,
        "torch_dtype_name": TORCH_DTYPE_NAME,
    }

    write_json(OUTPUT_DIR / "reward_config.json", config)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.set_tags({
            "stage": "step_rewards_exact",
            "model_short_name": MODEL_SHORT_NAME,
            "reward_step": "exact_action_match_0_1",
            "reward_final": "sympy_0_1",
            "srl_phase": "reward_table_before_training",
        })

        mlflow.log_params(config)
        mlflow.log_artifact(str(OUTPUT_DIR / "reward_config.json"), artifact_path="configs")

        print("[load] step dataset")
        step_df = pd.read_csv(STEP_DATA_PATH)
        validate_step_data(step_df)

        print("[load] model")
        tokenizer, model = load_train_model_and_tokenizer(
            model_name=MODEL_PATH,
            device=DEVICE,
            torch_dtype_name=TORCH_DTYPE_NAME,
            local_files_only=True,
        )

        print("[generate] model next steps")
        pred_df = generate_model_steps(
            df=step_df,
            model=model,
            tokenizer=tokenizer,
        )

        step_predictions_path = OUTPUT_DIR / "step_predictions.csv"
        pred_df.to_csv(step_predictions_path, index=False)
        mlflow.log_artifact(str(step_predictions_path), artifact_path="predictions")

        print("[reward] exact action rewards")
        rewards_df = compute_step_rewards(pred_df)

        print("[reward] final SymPy rewards")
        rewards_df = compute_final_sympy_rewards(rewards_df)

        rewards_path = OUTPUT_DIR / "step_rewards.csv"
        rewards_df.to_csv(rewards_path, index=False)
        mlflow.log_artifact(str(rewards_path), artifact_path="rewards")

        summary_df = make_summary(rewards_df)
        summary_path = OUTPUT_DIR / "reward_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        mlflow.log_artifact(str(summary_path), artifact_path="summary")

        for _, row in summary_df.iterrows():
            mlflow.log_metric(row["metric"], float(row["value"]))

        by_type = (
            rewards_df
            .groupby("type")["step_reward_exact"]
            .mean()
            .reset_index()
            .rename(columns={"step_reward_exact": "mean_step_reward_exact"})
        )
        by_type_path = OUTPUT_DIR / "reward_by_type.csv"
        by_type.to_csv(by_type_path, index=False)
        mlflow.log_artifact(str(by_type_path), artifact_path="summary")

        by_step = (
            rewards_df
            .groupby("step_id")["step_reward_exact"]
            .mean()
            .reset_index()
            .rename(columns={"step_reward_exact": "mean_step_reward_exact"})
        )
        by_step_path = OUTPUT_DIR / "reward_by_step_id.csv"
        by_step.to_csv(by_step_path, index=False)
        mlflow.log_artifact(str(by_step_path), artifact_path="summary")

        bad_examples = rewards_df[rewards_df["step_reward_exact"] == 0].head(100)
        bad_examples_path = OUTPUT_DIR / "bad_step_examples.csv"
        bad_examples.to_csv(bad_examples_path, index=False)
        mlflow.log_artifact(str(bad_examples_path), artifact_path="examples")

        good_examples = rewards_df[rewards_df["step_reward_exact"] == 1].head(100)
        good_examples_path = OUTPUT_DIR / "good_step_examples.csv"
        good_examples.to_csv(good_examples_path, index=False)
        mlflow.log_artifact(str(good_examples_path), artifact_path="examples")

        print("\nDone.")
        print(f"Outputs saved to: {OUTPUT_DIR}")
        print(summary_df)


if __name__ == "__main__":
    main()
