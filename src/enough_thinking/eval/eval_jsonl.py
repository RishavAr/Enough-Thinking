
import argparse, json, re
from pathlib import Path
from statistics import mean
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

def extract_answer_robust(text: str):
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        nums = re.findall(r"-?\d+", m.group(1))
        if nums:
            return int(nums[-1])
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None

def format_compliance(text: str):
    has_think = bool(re.search(r"<think>", text, re.IGNORECASE)) and bool(re.search(r"</think>", text, re.IGNORECASE))
    has_answer = bool(re.search(r"<answer>", text, re.IGNORECASE)) and bool(re.search(r"</answer>", text, re.IGNORECASE))
    return int(has_think and has_answer)

def load_jsonl(path: str, n: int | None = None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if n is not None and len(rows) >= n:
                break
    return rows

def build_prompt(mode: str, question: str):
    if mode == "plain":
        return f"Solve the following problem.\n\nQuestion: {question}\n\nAnswer:"
    if mode == "tagged_strict":
        return (
            "Return ONLY the following XML tags and nothing else.\n"
            "If you violate the format, your answer is wrong.\n\n"
            "<think>...</think>\n"
            "<answer>...</answer>\n\n"
            "Rules:\n"
            "- Put your reasoning inside <think>.\n"
            "- Put ONLY the final integer inside <answer>.\n\n"
            f"Question: {question}\n"
        )
    if mode == "easy_short":
        return (
            "You must follow this format exactly:\n"
            "<think>...</think>\n"
            "<answer>...</answer>\n\n"
            "Efficiency rule:\n"
            "- If the problem is EASY (1–2 steps), keep <think> extremely short (<= 1 line).\n"
            "- If EASY, do NOT write long reasoning. Answer quickly.\n"
            "- Put ONLY the final integer in <answer>.\n\n"
            f"Question: {question}\n"
        )
    raise ValueError("unknown mode")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--lora", default=None, help="Path to LoRA adapter (optional)")
    ap.add_argument("--data", required=True, help="Path to JSONL")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--mode", choices=["plain", "tagged_strict", "easy_short"], default="plain")
    args = ap.parse_args()

    rows = load_jsonl(args.data, n=args.n)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = base
    if args.lora is not None:
        if PeftModel is None:
            raise RuntimeError("peft not installed, but --lora was provided.")
        model = PeftModel.from_pretrained(base, args.lora)
    model.eval()

    acc, toks, comp = [], [], []

    for r in rows:
        q = r["prompt"]
        gold = int(r["answer"])

        prompt = build_prompt(args.mode, q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False
            )

        gen_tokens = out.shape[-1] - inputs["input_ids"].shape[-1]
        text = tokenizer.decode(out[0], skip_special_tokens=True)

        pred = extract_answer_robust(text)
        acc.append(1 if (pred is not None and pred == gold) else 0)
        toks.append(gen_tokens)
        comp.append(format_compliance(text))

    print({
        "n": len(rows),
        "mode": args.mode,
        "lora": args.lora,
        "accuracy": mean(acc),
        "avg_gen_tokens": mean(toks),
        "format_compliance_rate": mean(comp),
        "min_gen_tokens": min(toks),
        "max_gen_tokens": max(toks),
    })

if __name__ == "__main__":
    main()
