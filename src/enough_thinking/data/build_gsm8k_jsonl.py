
import argparse, json, re, random
from pathlib import Path
from datasets import load_dataset

def normalize_gold(ans: str):
    # GSM8K format: "... #### 18"
    m = re.search(r"####\s*(-?\d+)", ans)
    return m.group(1) if m else None

def estimate_steps(question: str):
    """
    Simple heuristic for "easy vs hard":
    - Count occurrences of numbers + operator words
    - More signals => likely multi-step
    This is intentionally lightweight; SEAL later will use this split for token penalties.
    """
    nums = re.findall(r"\d+", question)
    ops = re.findall(r"\b(total|left|remain|after|then|each|every|per|times|more|less|difference|sum)\b", question.lower())
    return len(nums) + len(ops)

def make_record(idx, split_name, q, gold):
    return {
        "id": f"gsm8k_{split_name}_{idx}",
        "prompt": q.strip(),
        "answer": gold,                 # integer as string
        "difficulty": None              # filled later
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_size", type=int, default=2000)
    ap.add_argument("--easy_threshold", type=int, default=8)  # lower => fewer "easy"
    args = ap.parse_args()

    random.seed(args.seed)

    ds = load_dataset("gsm8k", "main")
    train = ds["train"]
    test = ds["test"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build train subset
    train_indices = list(range(len(train)))
    random.shuffle(train_indices)
    train_indices = train_indices[:args.train_size]

    train_records = []
    for i, idx in enumerate(train_indices):
        q = train[idx]["question"]
        gold = normalize_gold(train[idx]["answer"])
        if gold is None:
            continue
        rec = make_record(i, "train", q, gold)
        steps = estimate_steps(q)
        rec["difficulty"] = "easy" if steps <= args.easy_threshold else "hard"
        train_records.append(rec)

    # Build eval sets from test
    easy_eval, hard_eval = [], []
    for i in range(len(test)):
        q = test[i]["question"]
        gold = normalize_gold(test[i]["answer"])
        if gold is None:
            continue
        rec = make_record(i, "test", q, gold)
        steps = estimate_steps(q)
        rec["difficulty"] = "easy" if steps <= args.easy_threshold else "hard"
        (easy_eval if rec["difficulty"] == "easy" else hard_eval).append(rec)

    # Save JSONL
    def save_jsonl(path: Path, records):
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    save_jsonl(out_dir / "math_train.jsonl", train_records)
    save_jsonl(out_dir / "math_eval_easy.jsonl", easy_eval)
    save_jsonl(out_dir / "math_eval_hard.jsonl", hard_eval)

    print("Wrote:")
    print(" -", out_dir / "math_train.jsonl", "records:", len(train_records))
    print(" -", out_dir / "math_eval_easy.jsonl", "records:", len(easy_eval))
    print(" -", out_dir / "math_eval_hard.jsonl", "records:", len(hard_eval))

if __name__ == "__main__":
    main()
