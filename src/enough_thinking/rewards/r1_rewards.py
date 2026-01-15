
import re

def extract_answer_robust(text: str):
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        nums = re.findall(r"-?\d+", m.group(1))
        if nums:
            return int(nums[-1])
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None

def format_ok(text: str) -> bool:
    has_think = bool(re.search(r"<think>", text, re.IGNORECASE)) and bool(re.search(r"</think>", text, re.IGNORECASE))
    has_answer = bool(re.search(r"<answer>", text, re.IGNORECASE)) and bool(re.search(r"</answer>", text, re.IGNORECASE))
    return has_think and has_answer

def r1_reward(text: str, gold_int: int, format_bonus: float = 0.2) -> float:
    pred = extract_answer_robust(text)
    correct = 1.0 if (pred is not None and pred == gold_int) else 0.0
    bonus = format_bonus if format_ok(text) else 0.0
    return correct + bonus
