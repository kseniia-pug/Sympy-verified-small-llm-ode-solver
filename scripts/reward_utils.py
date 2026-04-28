import re
from typing import Optional


def normalize_action(action: str) -> str:
    action = str(action).strip().lower()
    action = re.sub(r"[^a-z0-9_]+", "_", action)
    action = re.sub(r"_+", "_", action).strip("_")
    return action


def extract_action(step_text: str) -> str:
    text = str(step_text).strip()

    if text.startswith("FINAL:"):
        return "final_answer"

    match = re.search(r"ACTION\s+\d+:\s*(.*?)(?:\n|$)", text, re.DOTALL)
    if match:
        return normalize_action(match.group(1))

    return normalize_action(text.splitlines()[0] if text else "")


def action_exact_reward(model_step: str, target_action: str) -> int:
    model_action = extract_action(model_step)
    target_action = normalize_action(target_action)

    if target_action == "final_answer":
        return int(model_action == "final_answer")

    return int(model_action == target_action)


def build_next_step_prompt(equation: str, previous_steps: str) -> str:
    previous_steps = str(previous_steps).strip()

    if previous_steps == "" or previous_steps.lower() == "nan":
        previous_block = "<empty>"
    else:
        previous_block = previous_steps

    return f"""Solve the differential equation: {equation}

Previous solution steps:
{previous_block}

Generate only the next step of the expert solution trajectory.
Use this format:
ACTION k: ...
REASON k: ...

If this is the final step, use:
FINAL: \\boxed{{...}}
"""
