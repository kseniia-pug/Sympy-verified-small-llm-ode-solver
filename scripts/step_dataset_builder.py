import re
import pandas as pd
from pathlib import Path

def split_into_steps(trajectory: str) -> list:
    """Разбивает экспертную траекторию на отдельные шаги по ACTION N: или FINAL:"""
    if not isinstance(trajectory, str):
        return []
    pattern = r'(?=(?:ACTION \d+:|FINAL:))'
    parts = re.split(pattern, trajectory)
    steps = [part.strip() for part in parts if part.strip()]
    return steps

def extract_action(step_text: str) -> str:
    """Извлекает действие после ACTION N: (до конца строки)."""
    match = re.search(r'ACTION \d+:\s*(.*?)(?:\n|$)', step_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Если не ACTION, то вернём весь шаг (для fallback)
    return step_text.strip()

def make_step_prompt(equation: str, previous_steps: str) -> str:
    """Формирует вход для модели: уравнение + предыдущие шаги."""
    prev_text = previous_steps if previous_steps else "<empty>"
    return f"""Solve the differential equation: {equation}

Previous solution steps:
{prev_text}

Predict the next expert action and reason.
"""

def build_step_dataset(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        print(f"Пропущен (не найден): {input_path}")
        return

    df = pd.read_csv(input_path)
    required = {'equation', 'answer', 'type', 'expert_trajectory'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} не хватает колонок: {missing}")

    df = df[df['type'] != 'any'].copy()
    df.reset_index(drop=True, inplace=True)

    rows = []
    for idx, row in df.iterrows():
        eq = row['equation']
        ans = row['answer']
        typ = row['type']
        full_traj = row['expert_trajectory']
        steps = split_into_steps(full_traj)
        if not steps:
            print(f"Предупреждение: нет шагов в строке {idx}")
            continue

        example_id = row.get('example_id', f"{output_path.stem}_{idx}")
        previous = ""
        for step_num, step_text in enumerate(steps, start=1):
            # Определяем target_action
            if step_text.startswith("FINAL:"):
                target_action = "final_answer"
            else:
                target_action = extract_action(step_text)

            step_prompt = make_step_prompt(eq, previous)

            rows.append({
                'example_id': example_id,
                'step_id': step_num,
                'equation': eq,
                'type': typ,
                'previous_steps': previous,
                'step_prompt': step_prompt,
                'target_step': step_text,
                'target_action': target_action,
                'full_expert_trajectory': full_traj,
                'answer': ans,
            })
            # Добавляем текущий шаг к предыдущим для следующей итерации
            if previous:
                previous += "\n\n"
            previous += step_text

    if not rows:
        raise ValueError(f"{output_path} получился пустым (нет шагов ни в одном примере)")

    out_df = pd.DataFrame(rows)

    # Проверки
    if out_df["target_step"].isna().any():
        raise ValueError(f"Есть пустые target_step в {output_path}")
    # Проверим, что каждый target_step начинается с ACTION или FINAL
    starts_ok = out_df["target_step"].str.match(r'^\s*(ACTION|FINAL)').all()
    if not starts_ok:
        raise ValueError(f"Есть target_step без ACTION или FINAL в {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Создан {output_path}: {len(out_df)} строк (из {len(df)} исходных)")
    if not out_df.empty:
        print("Распределение по типам:\n", out_df['type'].value_counts(), "\n")
    else:
        print("Внимание: выходной DataFrame пуст!")

def main():
    data_dir = Path("data/reasoning_data")
    out_dir = Path("data/step_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [
        (data_dir / "mini_debug_reasoning_v1.csv", out_dir / "mini_debug_steps_v1.csv"),
        (data_dir / "balanced_train_small_reasoning_v1.csv", out_dir / "balanced_train_small_steps_v1.csv"),
        (data_dir / "balanced_val_small_reasoning_v1.csv", out_dir / "balanced_val_small_steps_v1.csv"),
        (data_dir / "mini_test_reasoning_v1.csv", out_dir / "mini_test_steps_v1.csv"),
        (data_dir / "train_big_reasoning_v1.csv", out_dir / "train_big_steps_v1.csv"),
        (data_dir / "val_big_reasoning_v1.csv", out_dir / "val_big_steps_v1.csv"),
        (data_dir / "balanced_val_metrics_big_reasoning_v1.csv", out_dir / "balanced_val_metrics_big_steps_v1.csv"),
    ]

    for in_path, out_path in files:
        build_step_dataset(in_path, out_path)

    # Сохраним метаданные
    metadata = {
        "description": "Step-wise dataset from expert trajectories (one row per step)",
        "version": "v1",
        "source": "template_by_type_from_cheatsheet",
        "columns": ["example_id", "step_id", "equation", "type", "previous_steps",
                    "step_prompt", "target_step", "target_action",
                    "full_expert_trajectory", "answer"],
        "files": [str(p) for _, p in files]
    }
    import json
    (out_dir / "step_dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Метаданные сохранены в", out_dir / "step_dataset_metadata.json")

if __name__ == "__main__":
    main()
