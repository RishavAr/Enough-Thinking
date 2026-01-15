# enough-thinking/src/enough_thinking/phase2_seal/run_phase2.py
import argparse, json, random, re
from pathlib import Path
from statistics import mean

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# -------------------------
# I/O
# -------------------------
def load_jsonl(path, n=None):
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
            if n and len(rows) >= n:
                break
    return rows


# -------------------------
# Prompts / parsing
# -------------------------
def extract_answer(text):
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S)
    if m:
        nums = re.findall(r"-?\d+", m.group(1))
        if nums:
            return int(nums[-1])
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None


def build_prompt(q, mode):
    if mode == "easy_short":
        return (
            "<think>short</think>\n"
            "<answer></answer>\n\n"
            f"Question: {q}\n"
        )
    return f"Question: {q}\n"


# -------------------------
# Evaluation
# -------------------------
def eval_model(model, tokenizer, rows):
    acc, toks = [], []
    for r in rows:
        p = build_prompt(r["prompt"], "easy_short")
        inp = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=128, do_sample=False)
        gen_tok = out.shape[-1] - inp["input_ids"].shape[-1]
        txt = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = extract_answer(txt)
        acc.append(1 if pred == int(r["answer"]) else 0)
        toks.append(gen_tok)
    return mean(acc), mean(toks)


# -------------------------
# Self-edit helpers
# -------------------------
def fallback_edit(q, a):
    return {
        "rule": "For easy arithmetic, compute directly and answer with minimal reasoning.",
        "pairs": [(q, str(a))]
    }


def inner_update_lora(model_name, tokenizer, lora_path, edit, steps=6):
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, lora_path, is_trainable=True)
    model.train()
    opt = AdamW(model.parameters(), lr=5e-4)

    for _ in range(steps):
        for q, a in edit["pairs"]:
            text = build_prompt(q, "easy_short") + f"<answer>{a}</answer>"
            tok = tokenizer(text, return_tensors="pt").to(model.device)
            loss = model(**tok, labels=tok["input_ids"]).loss
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    return model


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--phase1_lora", required=True)
    ap.add_argument("--easy_eval", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--outer_steps", type=int, default=5)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    base_for_eval = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    phase1 = PeftModel.from_pretrained(base_for_eval, args.phase1_lora)
    phase1.eval()

    rows = load_jsonl(args.easy_eval, n=60)
    base_acc, base_tok = eval_model(phase1, tokenizer, rows)
    print({"baseline_acc": base_acc, "baseline_tok": base_tok})

    out = []
    for step in range(1, args.outer_steps + 1):
        ex = random.choice(rows)
        edit = fallback_edit(ex["prompt"], ex["answer"])

        adapted = inner_update_lora(args.model, tokenizer, args.phase1_lora, edit)
        acc, tok = eval_model(adapted, tokenizer, rows)

        # ---------- SAFETY GATE ----------
        if acc < base_acc - 0.02:
            reward = -999.0
        else:
            reward = acc - 0.004 * tok
        # ---------------------------------

        print({"outer_step": step, "reward": reward, "acc": acc, "tok": tok})
        out.append({"reward": reward, "acc": acc, "tok": tok, "edit": edit})

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "accepted_edits.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    print("✅ Phase-2 with safety complete")


if __name__ == "__main__":
    main()
