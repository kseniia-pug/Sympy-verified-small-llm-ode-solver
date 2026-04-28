from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
INFERENCE модели + парсинг
"""


def build_prompt(equation: str) -> str:
    # Prompt for Qwen-2.5-math-1.5B
    return f"Solve the differential equation: {equation}. Put the final answer inside \\boxed{{}}."

def get_optimal_dtype():
    """Определяет оптимальный torch.dtype для текущего GPU."""
    if not torch.cuda.is_available():
        return torch.float32  # на CPU только float32

    # Проверяем поддержку bfloat16
    if torch.cuda.is_bf16_supported():
        print("Используется torch.bfloat16 (оптимально для A100/H100/H200)")
        return torch.bfloat16
    else:
        # V100 не поддерживает bf16, используем float16
        print("Используется torch.float16 (для V100)")
        return torch.float16

def load_model_and_tokenizer(
    model_name: str,
    device: str,
    torch_dtype = None,
    local_files_only: bool = False, #Берем с HF
):

    model_kwargs = {
        "local_files_only": local_files_only,
        "trust_remote_code": True,
    }
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=model_kwargs["local_files_only"],
        trust_remote_code=model_kwargs["trust_remote_code"]
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch_dtype

    if torch_dtype is None:
        dtype = get_optimal_dtype()
    else:
        dtype = torch_dtype
        
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    ).to(device)

    return tokenizer, model

def parse_final_answer(model_solution):
    if model_solution is None:
        return {
            "parsed_answer": None,
            "parse_success": 0,
        }

    text = str(model_solution)
    marker = r"\boxed{"
    last_answer = None
    start = 0

    #Эффективнее регулярных выражений для сложных (вложенных) latex конструкций
    while True:
        idx = text.find(marker, start)
        if idx == -1:
            break

        i = idx + len(marker)
        depth = 1

        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1

        if depth == 0:
            last_answer = text[idx + len(marker): i - 1].strip()
            start = i
        else:
            break

    return {
        "parsed_answer": last_answer,
        "parse_success": int(last_answer is not None),
    }


def run_batched_inference(
    df: pd.DataFrame,
    model,
    tokenizer,
    device: str,
    equation_col: str = "equation",
    batch_size: int = 4,
    max_new_tokens: int = 1536,
    do_samples: bool = False,
    save_every_batches: Optional[int] = None,
    partial_save_path: Optional[str] = None,
) -> pd.DataFrame:
    
    df = df.copy()
    df["prompt"] = df[equation_col].astype(str).apply(build_prompt)

    prompts = df["prompt"].tolist()
    all_outputs = []

    model.eval()
    tokenizer.padding_side = "left"

    batch_counter = 0
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]

        
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False # используем не чат модель!!!!
        ).to(device)
        
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_samples,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, prompt_len:]
        batch_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        all_outputs.extend([text.strip() for text in batch_texts])
        batch_counter += 1

        if save_every_batches is not None and partial_save_path is not None:
            if batch_counter % save_every_batches == 0:
                temp_df = df.iloc[:len(all_outputs)].copy()
                temp_df["llm_solution"] = all_outputs
                parse_results = temp_df["llm_solution"].apply(parse_final_answer)
                temp_df["parsed_answer"] = parse_results.apply(lambda x: x["parsed_answer"])
                temp_df["parse_success"] = parse_results.apply(lambda x: x["parse_success"])
                Path(partial_save_path).parent.mkdir(parents=True, exist_ok=True)
                temp_df.to_csv(partial_save_path, index=False)

    df["llm_solution"] = all_outputs

    parse_results = df["llm_solution"].apply(parse_final_answer)
    df["parsed_answer"] = parse_results.apply(lambda x: x["parsed_answer"])
    df["parse_success"] = parse_results.apply(lambda x: x["parse_success"])

    return df
