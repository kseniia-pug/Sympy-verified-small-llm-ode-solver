import json
from pathlib import Path
from textwrap import dedent

import pandas as pd


def ensure_boxed(answer: str) -> str:
    answer = str(answer).strip()
    if answer.startswith("\\boxed{") and answer.endswith("}"):
        return answer
    return f"\\boxed{{{answer}}}"


def build_prompt(equation: str) -> str:
    equation = str(equation).strip()
    return dedent(f"""
    Solve the differential equation: {equation}

    Produce an expert solution trajectory using ACTION and REASON fields.
    End with FINAL: \\boxed{{...}}.
    """).strip()


def separable_variables_actions(row) -> str:
    answer = ensure_boxed(row["answer"])
    return dedent(f"""
    ACTION 1: classify_equation
    REASON 1: The equation belongs to the separable variables class.

    ACTION 2: choose_method
    REASON 2: Use separation of variables.

    ACTION 3: separate_variables
    REASON 3: Rewrite the equation so that all y-dependent terms are on one side and all x-dependent terms are on the other side.

    ACTION 4: integrate_both_sides
    REASON 4: Integrate both sides and introduce an arbitrary constant.

    ACTION 5: construct_final_solution
    REASON 5: Express the integrated relation as the general solution.

    FINAL: {answer}
    """).strip()


def polynomial_action(row) -> str:
    answer = ensure_boxed(row["answer"])
    return dedent(f"""
    ACTION 1: classify_equation
    REASON 1: The equation is a first-order differential equation of the form y' = P(x), where P(x) is a polynomial.

    ACTION 2: choose_method
    REASON 2: Solve the equation by integrating the polynomial right-hand side with respect to x.

    ACTION 3: integrate_polynomial_terms
    REASON 3: Integrate each polynomial term separately using the power rule.

    ACTION 4: add_integration_constant
    REASON 4: Add an arbitrary constant C after integration.

    ACTION 5: construct_final_solution
    REASON 5: Write the antiderivative as the general solution.

    FINAL: {answer}
    """).strip()


def homogenous_2nd_order_action(row) -> str:
    answer = ensure_boxed(row["answer"])
    return dedent(f"""
    ACTION 1: classify_equation
    REASON 1: The equation is a linear homogeneous second-order differential equation.

    ACTION 2: choose_method
    REASON 2: Use the characteristic equation method.

    ACTION 3: build_characteristic_equation
    REASON 3: Replace y, y', and y'' by powers of the characteristic parameter to obtain the auxiliary equation.

    ACTION 4: solve_characteristic_equation
    REASON 4: Find the characteristic roots and determine whether they are distinct real roots, repeated roots, or complex conjugate roots.

    ACTION 5: construct_general_solution
    REASON 5: Build the complementary general solution according to the root case.

    FINAL: {answer}
    """).strip()


def inhomogenous_2nd_order_action(row) -> str:
    answer = ensure_boxed(row["answer"])
    return dedent(f"""
    ACTION 1: classify_equation
    REASON 1: The equation is a linear non-homogeneous second-order differential equation.

    ACTION 2: decompose_solution
    REASON 2: Represent the general solution as the sum of a homogeneous solution and a particular solution.

    ACTION 3: solve_homogeneous_part
    REASON 3: Solve the associated homogeneous equation using the characteristic equation method.

    ACTION 4: choose_particular_solution_method
    REASON 4: Choose a method for the particular solution, such as undetermined coefficients or variation of parameters, depending on the forcing term.

    ACTION 5: construct_particular_solution
    REASON 5: Find a particular solution that matches the non-homogeneous term.

    ACTION 6: combine_solutions
    REASON 6: Add the homogeneous solution and the particular solution to obtain the general solution.

    FINAL: {answer}
    """).strip()


def homogenous_3nd_order_action(row) -> str:
    answer = ensure_boxed(row["answer"])
    return dedent(f"""
    ACTION 1: classify_equation
    REASON 1: The equation is a linear homogeneous third-order differential equation.

    ACTION 2: choose_method
    REASON 2: Use the characteristic equation method for a homogeneous linear equation with constant coefficients.

    ACTION 3: build_characteristic_equation
    REASON 3: Replace y, y', y'', and y''' by powers of the characteristic parameter to obtain a cubic auxiliary equation.

    ACTION 4: solve_characteristic_equation
    REASON 4: Find the three characteristic roots, accounting for distinct, repeated, or complex roots.

    ACTION 5: construct_general_solution
    REASON 5: Build the homogeneous general solution from the characteristic roots.

    FINAL: {answer}
    """).strip()


def inhomogenous_3nd_order_action(row) -> str:
    answer = ensure_boxed(row["answer"])
    return dedent(f"""
    ACTION 1: classify_equation
    REASON 1: The equation is a linear non-homogeneous third-order differential equation.

    ACTION 2: decompose_solution
    REASON 2: Represent the general solution as the sum of a homogeneous solution and a particular solution.

    ACTION 3: solve_homogeneous_part
    REASON 3: Solve the associated homogeneous third-order equation using the characteristic equation method.

    ACTION 4: choose_particular_solution_method
    REASON 4: Choose a method for the particular solution based on the non-homogeneous forcing term.

    ACTION 5: construct_particular_solution
    REASON 5: Find a particular solution that satisfies the full non-homogeneous equation.

    ACTION 6: combine_solutions
    REASON 6: Add the homogeneous solution and the particular solution to obtain the general solution.

    FINAL: {answer}
    """).strip()


FUNCTIONS_MAP = {
    "separable variables": separable_variables_actions,
    "polynomial": polynomial_action,

    "homogenous 2nd order": homogenous_2nd_order_action,
    "homogeneous 2nd order": homogenous_2nd_order_action,

    "inhomogenous 2nd order": inhomogenous_2nd_order_action,
    "inhomogeneous 2nd order": inhomogenous_2nd_order_action,

    "homogenous 3nd order": homogenous_3nd_order_action,
    "homogeneous 3rd order": homogenous_3nd_order_action,

    "inhomogenous 3nd order": inhomogenous_3nd_order_action,
    "inhomogeneous 3rd order": inhomogenous_3nd_order_action,
}


def make_example_id_column(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    if "example_id" in df.columns:
        df = df.drop(columns=["example_id"])
    df.insert(0, "example_id", [f"{prefix}_{i:06d}" for i in range(len(df))])
    return df


def apply_logic(row):
    equation_type = str(row["type"]).strip()
    func = FUNCTIONS_MAP.get(equation_type)
    if func is None:
        return None
    return func(row)


def build_reasoning_dataset(input_path: str | Path, output_path: str | Path) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"[skip] Файл не найден: {input_path}")
        return

    df = pd.read_csv(input_path)

    required_cols = {"equation", "answer", "type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} не содержит обязательные колонки: {sorted(missing)}")

    df = df.copy()
    df["equation"] = df["equation"].astype(str)
    df["answer"] = df["answer"].astype(str)
    df["type"] = df["type"].astype(str).str.strip()

    df = df[df["type"] != "any"].copy()
    df = df.reset_index(drop=True)

    prefix = output_path.stem
    df = make_example_id_column(df, prefix=prefix)

    df["expert_trajectory"] = df.apply(apply_logic, axis=1)

    missing_mask = df["expert_trajectory"].isna()
    if missing_mask.any():
        missing_types = df.loc[missing_mask, "type"].value_counts()
        raise ValueError(f"Есть неподдержанные типы в {input_path}:\n{missing_types}")

    df["prompt"] = df["equation"].apply(build_prompt)
    df["target"] = df["expert_trajectory"]
    df["trajectory_version"] = "v1"
    df["trajectory_source"] = "template_by_type_from_cheatsheet"

    priority_cols = [
        "example_id",
        "equation",
        "answer",
        "type",
        "prompt",
        "expert_trajectory",
        "target",
        "trajectory_version",
        "trajectory_source",
    ]
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[priority_cols + other_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[saved] {output_path} ({len(df)} строк)")
    print(df["type"].value_counts())
    print()


def save_dataset_metadata(out_dir: Path) -> None:
    config = {
        "trajectory_version": "v1",
        "trajectory_source": "template_by_type_from_cheatsheet",
        "format": "ACTION_REASON_FINAL",
        "target_column": "expert_trajectory",
        "supported_types": sorted(FUNCTIONS_MAP.keys()),
        "note": "Reasoning datasets are generated both for small subsets and big train/val splits when files exist.",
    }

    config_path = out_dir / "dataset_build_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    notes = """# Reasoning dataset v1

This dataset uses template-based expert trajectories.

Each trajectory is written at the action level, not at the full formula-derivation level.

The final answer is taken from the original dataset and wrapped into `\\boxed{...}`.

Expected supervised learning format:

`prompt -> expert_trajectory`
"""

    notes_path = out_dir / "dataset_notes.md"
    notes_path.write_text(notes, encoding="utf-8")

    print(f"[saved] {config_path}")
    print(f"[saved] {notes_path}")


def main():
    out_dir = Path("data/reasoning_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [
        # small/debug files from make_balanced_split.py
        ("data/subsets/mini_debug.csv", "mini_debug_reasoning_v1.csv"),
        ("data/subsets/balanced_train_small.csv", "balanced_train_small_reasoning_v1.csv"),
        ("data/subsets/balanced_val_small.csv", "balanced_val_small_reasoning_v1.csv"),
        ("data/subsets/mini_test.csv", "mini_test_reasoning_v1.csv"),

        # big files from make_val_train_sets.py
        ("data/train_big.csv", "train_big_reasoning_v1.csv"),
        ("data/val_big.csv", "val_big_reasoning_v1.csv"),
        ("data/balanced_val_metrics_big.csv", "balanced_val_metrics_big_reasoning_v1.csv"),
    ]

    for input_path, output_name in files:
        build_reasoning_dataset(
            input_path=input_path,
            output_path=out_dir / output_name,
        )

    save_dataset_metadata(out_dir)


if __name__ == "__main__":
    main()
