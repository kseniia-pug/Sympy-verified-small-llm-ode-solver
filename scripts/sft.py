from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from inference_utils import build_prompt, run_batched_inference
from metrics_utils import compute_metrics
import mlflow


# -----------------------------
# dtype helpers
# -----------------------------


def get_optimal_dtype() -> torch.dtype:
    """Return the best dtype for the current device."""
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        print("Используется torch.bfloat16")
        return torch.bfloat16
    print("Используется torch.float16")
    return torch.float16


def choose_torch_dtype(dtype_name: str | torch.dtype | None = None) -> torch.dtype:
    """Convert a string dtype name to torch.dtype.

    Accepted values: None/"auto", "bfloat16", "float16", "float32".
    """
    if isinstance(dtype_name, torch.dtype):
        return dtype_name

    if dtype_name is None or dtype_name == "auto":
        return get_optimal_dtype()

    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unknown torch dtype: {dtype_name}")
    return mapping[dtype_name]


def _safe_metric_key(text: str) -> str:
    """Make metric names stable and MLflow-friendly."""
    text = str(text).strip().lower()
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def _maybe_float(value: Any) -> Optional[float]:
    """Return float(value) or None for NA/non-finite values."""
    try:
        if pd.isna(value):
            return None
        out = float(value)
        if not torch.isfinite(torch.tensor(out)):
            return None
        return out
    except Exception:
        return None


# -----------------------------
# data helpers
# -----------------------------


def make_target(answer: str) -> str:
    answer = str(answer).strip()
    if answer.startswith("\\boxed{") and answer.endswith("}"):
        return answer
    return f"\\boxed{{{answer}}}"


def make_full_text(equation: str, answer: str) -> str:
    return build_prompt(equation) + make_target(answer)


def load_equations_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    required_cols = {"equation", "answer", "type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["equation"] = df["equation"].astype(str)
    df["answer"] = df["answer"].astype(str)
    df["type"] = df["type"].astype(str)

    df["prompt"] = df["equation"].apply(build_prompt)
    df["target"] = df["answer"].apply(make_target)
    df["full_text"] = df.apply(lambda row: make_full_text(row["equation"], row["answer"]), axis=1)
    return df


def load_train_val_dataframes(train_path: str | Path, val_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return load_equations_csv(train_path), load_equations_csv(val_path)


def balanced_sample(
    df: pd.DataFrame,
    col: str = "type",
    n_per_type: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a fixed balanced sample with up to n_per_type rows per class."""
    sampled = df.groupby(col, group_keys=False).apply(
        lambda x: x.sample(min(len(x), n_per_type), random_state=random_state)
    )
    return sampled.reset_index(drop=True)


# -----------------------------
# tokenizer / model loading
# -----------------------------


def load_train_model_and_tokenizer(
    model_name: str,
    device: str,
    torch_dtype_name: str | torch.dtype | None = "auto",
    local_files_only: bool = False,
    trust_remote_code: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    dtype = choose_torch_dtype(torch_dtype_name)
    model_kwargs = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
    }

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)

    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


# -----------------------------
# parameter helpers
# -----------------------------


def freeze_all_except_lm_head(model) -> None:
    for param in model.parameters():
        param.requires_grad = False

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("Model does not expose output embeddings / lm head.")

    for param in output_embeddings.parameters():
        param.requires_grad = True


def unfreeze_all_parameters(model) -> None:
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def get_trainable_parameter_names(model) -> List[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]


# -----------------------------
# tokenization
# -----------------------------


def tokenize_sft_example(prompt: str, target: str, tokenizer) -> Dict[str, List[int]]:
    # Non-chat model: do not add extra special tokens here.
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids.copy()
    attention_mask = [1] * len(input_ids)

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class SFTDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.dataframe.iloc[idx]
        return tokenize_sft_example(prompt=row["prompt"], target=row["target"], tokenizer=self.tokenizer)


class CausalSFTCollator:
    def __init__(self, tokenizer, padding_side: str = "right"):
        if padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be 'left' or 'right'")
        self.tokenizer = tokenizer
        self.padding_side = padding_side

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)
        pad_id = self.tokenizer.pad_token_id

        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for x in features:
            pad_len = max_len - len(x["input_ids"])

            if self.padding_side == "left":
                input_ids = [pad_id] * pad_len + x["input_ids"]
                labels = [-100] * pad_len + x["labels"]
                attention_mask = [0] * pad_len + x["attention_mask"]
            else:
                input_ids = x["input_ids"] + [pad_id] * pad_len
                labels = x["labels"] + [-100] * pad_len
                attention_mask = x["attention_mask"] + [0] * pad_len

            batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            batch_labels.append(torch.tensor(labels, dtype=torch.long))
            batch_attention_mask.append(torch.tensor(attention_mask, dtype=torch.long))

        return {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "attention_mask": torch.stack(batch_attention_mask),
        }


# -----------------------------
# configs
# -----------------------------


@dataclass
class SFTConfig:
    output_dir: str
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    save_total_limit: int = 3
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    report_to: str | List[str] | None = "mlflow"
    remove_unused_columns: bool = False
    dataloader_num_workers: int = 0
    seed: int = 42
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch"
    max_grad_norm: float = 1.0
    save_safetensors: bool = True
    run_name: Optional[str] = None


@dataclass
class GenerationEvalConfig:
    batch_size: int = 8
    max_new_tokens: int = 1536
    do_sample: bool = False
    porog: float = 0.5
    compute_reference: bool = False
    torch_dtype_name: str = "bfloat16"
    local_files_only: bool = False
    device: str = "cuda"
    save_every_batches: int = 1
    log_every: int = 10
    save_every_rows: int = 25
    sympy_timeout_sec: int = 8
    sample_per_type: Optional[int] = None


@dataclass
class MLflowConfig:
    experiment_name: str = "ode_llm_diploma"
    run_name: str = "answersft_qwen25_math_15b_balanced_train_small_run01"
    log_code: bool = True
    log_final_model: bool = True


# -----------------------------
# trainer
# -----------------------------


def build_training_args(config: SFTConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        eval_strategy=config.eval_strategy,
        save_total_limit=config.save_total_limit,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to=config.report_to,
        remove_unused_columns=config.remove_unused_columns,
        dataloader_num_workers=config.dataloader_num_workers,
        seed=config.seed,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,
        max_grad_norm=config.max_grad_norm,
        save_safetensors=config.save_safetensors,
        run_name=config.run_name,
    )


def build_sft_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, tokenizer):
    train_dataset = SFTDataset(train_df, tokenizer=tokenizer)
    val_dataset = SFTDataset(val_df, tokenizer=tokenizer)
    return train_dataset, val_dataset


@torch.inference_mode()
def evaluate_sympy_metrics(
    model,
    tokenizer,
    val_df: pd.DataFrame,
    sample_per_type: Optional[int] = None,
    batch_size: int = 8,
    max_new_tokens: int = 1536,
    device: str = "cuda",
    timeout_sec: int = 8,
) -> dict:
    """Run a small generation+SymPy eval during training.

    This is intentionally small and balanced by class. It is a training-time
    diagnostic, not the final validation metric.
    """
    was_training = model.training
    original_padding_side = tokenizer.padding_side
    metrics_dict: Dict[str, float] = {}

    try:
        model.eval()
        tokenizer.padding_side = "left"

        # If sample_per_type is None, evaluate exactly the provided dataframe.
        # Use this mode when val_df is already a fixed balanced metrics set.
        if sample_per_type is None:
            sample_df = val_df.reset_index(drop=True)
        else:
            sample_df = balanced_sample(val_df, n_per_type=sample_per_type)

        result_df = run_batched_inference(
            df=sample_df,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            do_samples=False,
            save_every_batches=None,
            partial_save_path=None,
        )

        _, metrics_df, class_df = compute_metrics(
            data=result_df,
            porog=0.5,
            progress_log_path=None,
            log_every=10,
            partial_eval_save_path=None,
            save_every=25,
            compute_reference=False,
            timeout_sec=timeout_sec,
        )

        for _, row in metrics_df.iterrows():
            value = _maybe_float(row["value"])
            if value is not None:
                metrics_dict[str(row["metric"])] = value

        for _, row in class_df.iterrows():
            eq_type = _safe_metric_key(row["equation_type"])
            observed = _maybe_float(row["class_wise_pass_rate_observed"])
            strict = _maybe_float(row["class_wise_pass_rate_strict"])
            if observed is not None:
                metrics_dict[f"class_{eq_type}_observed"] = observed
            if strict is not None:
                metrics_dict[f"class_{eq_type}_strict"] = strict

    finally:
        tokenizer.padding_side = original_padding_side
        if was_training:
            model.train()
        else:
            model.eval()

    return metrics_dict


class CustomTrainer(Trainer):
    def __init__(
        self,
        val_df_for_metrics: Optional[pd.DataFrame] = None,
        sample_per_type: Optional[int] = None,
        generation_batch_size: int = 8,
        generation_max_new_tokens: int = 1536,
        sympy_timeout_sec: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.val_df_for_metrics = val_df_for_metrics
        self.sample_per_type = sample_per_type
        self.generation_batch_size = generation_batch_size
        self.generation_max_new_tokens = generation_max_new_tokens
        self.sympy_timeout_sec = sympy_timeout_sec

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        if self.val_df_for_metrics is not None:
            sympy_metrics = evaluate_sympy_metrics(
                model=self.model,
                tokenizer=self.tokenizer,
                val_df=self.val_df_for_metrics,
                sample_per_type=self.sample_per_type,
                batch_size=self.generation_batch_size,
                max_new_tokens=self.generation_max_new_tokens,
                device=str(self.args.device),
                timeout_sec=self.sympy_timeout_sec,
            )
            # Добавляем префикс "eval_"
            sympy_metrics = {f"{metric_key_prefix}_{k}": v for k, v in sympy_metrics.items()}
            
            # 1. Логируем в MLflow явно (с текущим шагом)
            if mlflow.active_run():
                mlflow.log_metrics(sympy_metrics, step=self.state.global_step)
            
            # 2. Вызываем стандартный self.log (для совместимости)
            self.log(sympy_metrics)
            
            # 3. Сохраняем метрики в CSV для offline анализа
            output_dir = Path(self.args.output_dir)
            metrics_file = output_dir / "eval_metrics_history.csv"
            record = {"step": self.state.global_step, **sympy_metrics}
            if metrics_file.exists():
                df = pd.read_csv(metrics_file)
                df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
            else:
                df = pd.DataFrame([record])
            df.to_csv(metrics_file, index=False)
            
            output.update(sympy_metrics)
        return output


def build_trainer(
    model,
    tokenizer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: SFTConfig,
    padding_side: str = "right",
    val_df_for_metrics: Optional[pd.DataFrame] = None,
    sample_per_type: Optional[int] = None,
    generation_batch_size: int = 8,
    generation_max_new_tokens: int = 1536,
    sympy_timeout_sec: int = 8,
) -> CustomTrainer:
    train_dataset, val_dataset = build_sft_datasets(train_df=train_df, val_df=val_df, tokenizer=tokenizer)
    collator = CausalSFTCollator(tokenizer=tokenizer, padding_side=padding_side)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    training_args = build_training_args(config)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        val_df_for_metrics=val_df_for_metrics,
        sample_per_type=sample_per_type,
        generation_batch_size=generation_batch_size,
        generation_max_new_tokens=generation_max_new_tokens,
        sympy_timeout_sec=sympy_timeout_sec,
    )
    return trainer


def train_sft(
    model,
    tokenizer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: SFTConfig,
    debug_freeze: bool = False,
    padding_side: str = "right",
    val_df_for_metrics: Optional[pd.DataFrame] = None,
    sample_per_type: Optional[int] = None,
    generation_batch_size: int = 8,
    generation_max_new_tokens: int = 1536,
    sympy_timeout_sec: int = 8,
):
    if debug_freeze:
        freeze_all_except_lm_head(model)
    else:
        unfreeze_all_parameters(model)

    model.train()

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_df=train_df,
        val_df=val_df,
        config=config,
        padding_side=padding_side,
        val_df_for_metrics=val_df_for_metrics,
        sample_per_type=sample_per_type,
        generation_batch_size=generation_batch_size,
        generation_max_new_tokens=generation_max_new_tokens,
        sympy_timeout_sec=sympy_timeout_sec,
    )
    train_result = trainer.train()
    return trainer, train_result


# -----------------------------
# saving
# -----------------------------


def save_model_and_tokenizer(
    trainer: Optional[Trainer] = None,
    model=None,
    tokenizer=None,
    save_dir: str | Path = "./sft_model",
) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if trainer is not None:
        trainer.save_model(str(save_dir))
        if tokenizer is None:
            tokenizer = trainer.tokenizer
    elif model is not None:
        model.save_pretrained(str(save_dir))
    else:
        raise ValueError("Pass either trainer or model.")

    if tokenizer is None:
        raise ValueError("Tokenizer must be provided or available from trainer.")

    tokenizer.save_pretrained(str(save_dir))
    return save_dir
