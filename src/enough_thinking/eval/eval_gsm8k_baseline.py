import os, re, argparse
from statistics import mean
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def normalize_gold(ans: str):
    m = re.search(r"####\s*(-?\d+)", ans)
    return int(m.group(1)) if m else None

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--mode", choices=["plain", "tagged_strict", "easy_short"], default="plain")
    args = ap.parse_args()

    gsm8k = load_dataset("gsm8k", "main")

    hf_token = os.environ.get("HF_TOKEN", None)

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=hf_token,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    acc, toks, comp = [], [], []

    for i in range(args.n):
        ex = gsm8k["test"][i]
        gold = normalize_gold(ex["answer"])

        if args.mode == "plain":
            prompt = f"Solve the following problem.\n\nQuestion: {ex['question']}\n\nAnswer:"
        elif args.mode == "tagged_strict":
            prompt = (
                "Return ONLY the following XML tags and nothing else.\n"
                "If you violate the format, your answer is wrong.\n\n"
                "<think>...</think>\n"
                "<answer>...</answer>\n\n"
                "Rules:\n"
                "- Put your reasoning inside <think>.\n"
                "- Put ONLY the final integer inside <answer>.\n\n"
                f"Question: {ex['question']}\n"
            )
        else:  # easy_short
            prompt = (
                "You must follow this format exactly:\n"
                "<think>...</think>\n"
                "<answer>...</answer>\n\n"
                "Efficiency rule:\n"
                "- If the problem is EASY (1–2 steps), keep <think> extremely short (<= 1 line).\n"
                "- If EASY, do NOT write long reasoning. Answer quickly.\n"
                "- Put ONLY the final integer in <answer>.\n\n"
                f"Question: {ex['question']}\n"
            )

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
        acc.append(1 if (pred is not None and gold is not None and pred == gold) else 0)
        toks.append(gen_tokens)
        comp.append(format_compliance(text))

    result = {
        "n": args.n,
        "mode": args.mode,
        "accuracy": mean(acc),
        "avg_gen_tokens": mean(toks),
        "format_compliance_rate": mean(comp),
        "min_gen_tokens": min(toks),
        "max_gen_tokens": max(toks),
    }
    print(result)

if __name__ == "__main__":
    main()
