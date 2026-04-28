from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import mlflow
import pandas as pd
import torch

from sft import (
    GenerationEvalConfig,
    SFTConfig,
    count_parameters,
    get_trainable_parameter_names,
    load_train_model_and_tokenizer,
    save_model_and_tokenizer,
    train_sft,
)

from sft_reasoning import (
    load_reasoning_csv,
    load_reasoning_train_val_dataframes,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
MODEL_SHORT_NAME = "qwen25_math_15b"

REASONING_VERSION = "v1"

TRAIN_NAME = "balanced_train_small_reasoning_v1"
VAL_NAME = "balanced_val_small_reasoning_v1"
VAL_METRICS_NAME = "balanced_val_small_reasoning_v1"

TRAIN_PATH = Path("data/reasoning_data/balanced_train_small_reasoning_v1.csv")
VAL_PATH = Path("data/reasoning_data/balanced_val_small_reasoning_v1.csv")
VAL_METRICS_PATH = Path("data/reasoning_data/balanced_val_small_reasoning_v1.csv")

BASE_OUTPUT_DIR = Path("experiments/reasoning_sft")
MLFLOW_EXPERIMENT = "ode_llm_diploma"

SEED = 42
SAMPLE_PER_TYPE_SOURCE = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    TORCH_DTYPE_NAME = "bfloat16"
elif torch.cuda.is_available():
    TORCH_DTYPE_NAME = "float16"
else:
    TORCH_DTYPE_NAME = "float32"


# ============================================================
# Helpers
# ============================================================

def next_run_dir(base_dir: Path, prefix: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)

    run_num = 1
    while (base_dir / f"{prefix}_run{run_num:02d}").exists():
        run_num += 1

    out = base_dir / f"{prefix}_run{run_num:02d}"
    out.mkdir(parents=True, exist_ok=True)

    return out


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def check_input_files() -> None:
    missing = []

    for path in [TRAIN_PATH, VAL_PATH, VAL_METRICS_PATH]:
        if not path.exists():
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Не найдены нужные reasoning-файлы:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\n\nСначала запусти:\n"
            + "  python scripts/reasoning_dataset_builder.py"
        )


def log_dataframe_info(df: pd.DataFrame, name: str) -> None:
    print(f"\n[{name}] shape={df.shape}")
    if "type" in df.columns:
        print(df["type"].value_counts())

    print("\nПример prompt:")
    print(df.iloc[0]["prompt"][:700])

    print("\nПример target:")
    print(df.iloc[0]["target"][:1200])


# ============================================================
# Main
# ============================================================

def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    check_input_files()

    run_prefix = f"{MODEL_SHORT_NAME}_{TRAIN_NAME}"
    run_dir = next_run_dir(BASE_OUTPUT_DIR, run_prefix)
    run_name = f"reasoningsft_{run_dir.name}"

    train_cfg = SFTConfig(
        output_dir=str(run_dir),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=3,
        bf16=(TORCH_DTYPE_NAME == "bfloat16"),
        fp16=(TORCH_DTYPE_NAME == "float16"),
        gradient_checkpointing=True,
        report_to="mlflow",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=SEED,
        run_name=run_name,
    )

    eval_cfg = GenerationEvalConfig(
        batch_size=4,
        max_new_tokens=2048,
        do_sample=False,
        porog=0.5,
        compute_reference=False,
        torch_dtype_name=TORCH_DTYPE_NAME,
        local_files_only=False,
        device=DEVICE,
        sympy_timeout_sec=8,

        # ВАЖНО:
        # metrics-val уже заранее фиксированный и balanced.
        # Поэтому внутри sft.py не надо делать новый sample.
        sample_per_type=None,
    )

    train_config_path = run_dir / "train_config.json"
    eval_config_path = run_dir / "eval_config.json"
    dataset_config_path = run_dir / "dataset_config.json"
    trainer_state_path = run_dir / "trainer_state.json"
    trainable_names_path = run_dir / "trainable_parameter_names.txt"

    write_json(train_config_path, asdict(train_cfg))
    write_json(eval_config_path, asdict(eval_cfg))
    write_json(
        dataset_config_path,
        {
            "stage": "reasoning_sft",
            "reasoning_version": REASONING_VERSION,

            "train_name": TRAIN_NAME,
            "train_path": str(TRAIN_PATH),

            "val_name": VAL_NAME,
            "val_path": str(VAL_PATH),

            "val_metrics_name": VAL_METRICS_NAME,
            "val_metrics_path": str(VAL_METRICS_PATH),

            "target_column": "expert_trajectory",
            "training_format": "prompt -> expert_trajectory",

            "val_metrics_is_fixed": True,
            "val_metrics_created_by": "reasoning_dataset_builder.py",
            "resample_during_training_eval": False,

            "sample_per_type_source": SAMPLE_PER_TYPE_SOURCE,
            "seed": SEED,
        },
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "stage": "reasoning_sft",
                "model_name": MODEL_NAME,
                "model_short_name": MODEL_SHORT_NAME,

                "dataset_train": TRAIN_NAME,
                "dataset_val": VAL_NAME,
                "dataset_val_metrics": VAL_METRICS_NAME,

                "reasoning_version": REASONING_VERSION,
                "target_column": "expert_trajectory",
                "split_scope": "big",
                "eval_protocol": "fixed_balanced_val_metrics_no_resampling",

                "output_dir": str(run_dir),
            }
        )

        mlflow.log_params(
            {
                "model_name": MODEL_NAME,

                "train_path": str(TRAIN_PATH),
                "val_path": str(VAL_PATH),
                "val_metrics_path": str(VAL_METRICS_PATH),

                "fixed_val_metrics": True,
                "training_eval_resample": False,
                "sample_per_type_source": SAMPLE_PER_TYPE_SOURCE,

                "device": DEVICE,
                "torch_dtype_name": TORCH_DTYPE_NAME,

                "num_train_epochs": train_cfg.num_train_epochs,
                "per_device_train_batch_size": train_cfg.per_device_train_batch_size,
                "per_device_eval_batch_size": train_cfg.per_device_eval_batch_size,
                "gradient_accumulation_steps": train_cfg.gradient_accumulation_steps,

                "learning_rate": train_cfg.learning_rate,
                "weight_decay": train_cfg.weight_decay,
                "warmup_ratio": train_cfg.warmup_ratio,

                "bf16": train_cfg.bf16,
                "fp16": train_cfg.fp16,
                "gradient_checkpointing": train_cfg.gradient_checkpointing,

                "seed": train_cfg.seed,

                "gen_batch_size": eval_cfg.batch_size,
                "gen_max_new_tokens": eval_cfg.max_new_tokens,
                "sympy_timeout_sec": eval_cfg.sympy_timeout_sec,
            }
        )

        # Логируем конфиги сразу.
        mlflow.log_artifact(str(train_config_path), artifact_path="configs")
        mlflow.log_artifact(str(eval_config_path), artifact_path="configs")
        mlflow.log_artifact(str(dataset_config_path), artifact_path="configs")
        mlflow.log_artifact(str(VAL_METRICS_PATH), artifact_path="datasets")

        print("[load] reasoning data")
        train_df, val_df = load_reasoning_train_val_dataframes(TRAIN_PATH, VAL_PATH)
        val_metrics_df = load_reasoning_csv(VAL_METRICS_PATH)

        log_dataframe_info(train_df, "train_df")
        log_dataframe_info(val_df, "val_df")
        log_dataframe_info(val_metrics_df, "fixed balanced val_metrics_df")

        print("\n[load] model")
        tokenizer, model = load_train_model_and_tokenizer(
            model_name=MODEL_NAME,
            device=DEVICE,
            torch_dtype_name=TORCH_DTYPE_NAME,
            local_files_only=False,
        )

        param_stats = count_parameters(model)
        mlflow.log_metrics({k: float(v) for k, v in param_stats.items()})

        trainable_names_path.write_text(
            "\n".join(get_trainable_parameter_names(model)),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(trainable_names_path), artifact_path="debug")

        print("\n[train] reasoning-SFT")
        trainer, train_result = train_sft(
            model=model,
            tokenizer=tokenizer,
            train_df=train_df,
            val_df=val_df,
            val_df_for_metrics=val_metrics_df,

            # ВАЖНО:
            # val_metrics_df уже готовый fixed balanced metrics set.
            sample_per_type=None,

            generation_batch_size=eval_cfg.batch_size,
            generation_max_new_tokens=eval_cfg.max_new_tokens,
            sympy_timeout_sec=eval_cfg.sympy_timeout_sec,

            config=train_cfg,
            debug_freeze=False,
            padding_side="right",
        )

        if train_result.metrics:
            mlflow.log_metrics(
                {
                    f"train_result_{k}": float(v)
                    for k, v in train_result.metrics.items()
                    if isinstance(v, (int, float))
                }
            )

        trainer.state.save_to_json(str(trainer_state_path))
        mlflow.log_artifact(str(trainer_state_path), artifact_path="trainer")

        eval_metrics_csv = run_dir / "eval_metrics_history.csv"
        if eval_metrics_csv.exists():
            mlflow.log_artifact(str(eval_metrics_csv), artifact_path="metrics")
            print(f"[MLflow] logged {eval_metrics_csv}")

        final_model_dir = save_model_and_tokenizer(
            trainer=trainer,
            tokenizer=tokenizer,
            save_dir=run_dir / "final_model",
        )

        mlflow.log_artifacts(str(final_model_dir), artifact_path="final_model")

        print("\nReasoning-SFT завершен.")
        print(f"Финальная модель: {final_model_dir}")
        print(f"MLflow run name: {run_name}")
        print(f"Run dir: {run_dir}")
        print("\nДальше отдельно запусти inference+metrics на final_model для val_small_reasoning_v1.")


if __name__ == "__main__":
    main()
